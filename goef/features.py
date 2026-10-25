"""DINOv3 image embeddings used for representative-image selection.

GOEF clusters *normal training images* in the embedding space of a frozen
visual encoder (Sec. III-D of the paper uses DINOv3).  This module contains the
encoder loading logic and the batched embedding extraction; the selection
algorithms themselves live in :mod:`goef.selection`.

The implementation accepts the public DINOv3 timm checkpoint and transparently
adapts the key layout used by ``facebookresearch/dinov3`` to the layout used by
``timm`` (layer-scale names, split q/v biases, storage/register token).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
DEFAULT_MODEL = "vit_base_patch16_dinov3.lvd1689m"


# --------------------------------------------------------------------------- #
# Image IO
# --------------------------------------------------------------------------- #
def list_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Return every image file below *directory* in deterministic order."""
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_transform(image_size: int) -> Callable[[Image.Image], "object"]:
    """Resize + ImageNet-normalise transform identical to the paper's protocol."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    def transform(image: Image.Image):
        import torch

        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(array)

    return transform


# --------------------------------------------------------------------------- #
# Checkpoint handling
# --------------------------------------------------------------------------- #
def _safe_torch_load(weight_path: Path):
    import torch

    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13
        return torch.load(weight_path, map_location="cpu")
    except Exception:
        return torch.load(weight_path, map_location="cpu", weights_only=False)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "teacher", "student", "backbone"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _clean_state_dict_keys(state_dict):
    prefixes = ("module.", "model.", "backbone.", "teacher.", "student.", "encoder.")
    cleaned = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def _convert_dinov3_timm_keys(state_dict):
    """Map an original DINOv3 checkpoint onto ``timm`` parameter names."""
    converted = dict(state_dict)
    if "storage_tokens" in converted and "reg_token" not in converted:
        converted["reg_token"] = converted.pop("storage_tokens")

    block_ids = {
        int(key.split(".")[1])
        for key in converted
        if key.startswith("blocks.")
        and len(key.split(".")) > 2
        and key.split(".")[1].isdigit()
    }
    for block_id in block_ids:
        for source, target in (
            (f"blocks.{block_id}.ls1.gamma", f"blocks.{block_id}.gamma_1"),
            (f"blocks.{block_id}.ls2.gamma", f"blocks.{block_id}.gamma_2"),
        ):
            if source in converted and target not in converted:
                converted[target] = converted.pop(source)

        qkv_key = f"blocks.{block_id}.attn.qkv.bias"
        q_key = f"blocks.{block_id}.attn.q_bias"
        v_key = f"blocks.{block_id}.attn.v_bias"
        if qkv_key in converted and q_key not in converted and v_key not in converted:
            qkv_bias = converted[qkv_key]
            if qkv_bias.ndim == 1 and qkv_bias.numel() % 3 == 0:
                hidden_dim = qkv_bias.numel() // 3
                converted.pop(qkv_key)
                converted[q_key] = qkv_bias[:hidden_dim]
                converted[v_key] = qkv_bias[2 * hidden_dim:]
    return converted


def load_encoder(weight_path: Path, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Build a DINOv3 timm encoder and load *weight_path* into it."""
    try:
        import timm
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "timm is required for representative selection: pip install timm"
        ) from error

    print(f"building encoder: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    state_dict = _convert_dinov3_timm_keys(
        _clean_state_dict_keys(_extract_state_dict(_safe_torch_load(weight_path)))
    )
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    if not matched:
        raise RuntimeError(
            f"no parameter of {weight_path} matches {model_name}; "
            "check --model-name and the checkpoint"
        )
    report = model.load_state_dict(matched, strict=False)
    print(
        f"loaded {len(matched)}/{len(model_state)} tensors "
        f"(missing={len(report.missing_keys)}, unexpected={len(report.unexpected_keys)})"
    )
    if len(matched) < len(model_state) * 0.7:
        print("WARNING: less than 70% of the encoder weights were matched")
    model.eval().to(device)
    return model


def pool_features(model, batch):
    """Return L2-normalised image-level embeddings for one input batch."""
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode():
        features = model(batch)
        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "pooled", "features"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise RuntimeError(f"unrecognised model output: {list(features)}")
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        return functional.normalize(features.float(), dim=1)


def extract_embeddings(
    model,
    image_paths: Sequence[Path],
    transform,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract one L2-normalised embedding per image, shape ``[N, D]``."""
    import torch
    from tqdm import tqdm

    if not image_paths:
        raise ValueError("no images to embed")
    batches = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="DINOv3 embeddings"):
        paths = image_paths[start:start + batch_size]
        try:
            batch = torch.stack([transform(load_image(path)) for path in paths]).to(device)
        except Exception as error:
            raise RuntimeError(f"failed to read image batch starting at {paths[0]}") from error
        batches.append(pool_features(model, batch).cpu().numpy())
    return np.concatenate(batches, axis=0)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def patchcore_random_projection(features: np.ndarray, seed: int, target_dim: int) -> np.ndarray:
    """PatchCore's optional random linear reduction, reproduced in NumPy.

    PatchCore uses an untrained bias-free ``torch.nn.Linear`` whose Kaiming
    uniform initialisation has bounds ``+-1/sqrt(input_dim)``.  Reproducing it
    here keeps the coreset selection deterministic and dependency free.
    """
    input_dim = features.shape[1]
    if target_dim <= 0 or input_dim <= target_dim:
        return features
    rng = np.random.default_rng(seed)
    bound = 1.0 / math.sqrt(input_dim)
    weights = rng.uniform(-bound, bound, size=(target_dim, input_dim))
    return features @ weights.T
