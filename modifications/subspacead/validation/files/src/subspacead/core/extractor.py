import logging
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from transformers import (
    AutoImageProcessor,
    AutoModel,
    DINOv3ViTConfig,
    DINOv3ViTImageProcessorFast,
    DINOv3ViTModel,
)
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """Encapsulates the feature extraction model and logic."""

    def __init__(self, model_ckpt: str, enable_attention_maps: bool = False):
        logging.info(f"Loading feature extraction model: {model_ckpt}...")
        self.enable_attention_maps = enable_attention_maps
        self.backend = "hf"
        if self._is_dinov3_checkpoint(model_ckpt):
            dinov3_backend = os.environ.get(
                "SUBSPACEAD_DINOV3_BACKEND", "transformers"
            ).lower()
            if dinov3_backend == "native":
                if self.enable_attention_maps:
                    raise ValueError(
                        "DINO saliency masking is not supported for native local DINOv3 .pth checkpoints."
                    )
                self.backend = "dinov3_native"
                self.processor = None
                self.model = self._load_dinov3_native_model(model_ckpt).eval().to(DEVICE)
                logging.info("Loaded local DINOv3 checkpoint with native backend.")
            elif dinov3_backend == "transformers":
                self.backend = "dinov3_transformers"
                self.processor, self.model = self._load_dinov3_transformers_model(
                    model_ckpt
                )
                logging.info(
                    "Loaded local DINOv3 checkpoint with Transformers backend."
                )
            else:
                raise ValueError(
                    "SUBSPACEAD_DINOV3_BACKEND must be 'transformers' or 'native'."
                )
        else:
            local_files_only = any(
                os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
                for name in (
                    "SUBSPACEAD_HF_LOCAL_FILES_ONLY",
                    "TRANSFORMERS_OFFLINE",
                    "HF_HUB_OFFLINE",
                )
            )
            self.processor = AutoImageProcessor.from_pretrained(
                model_ckpt, local_files_only=local_files_only
            )
            self.model = (
                AutoModel.from_pretrained(
                    model_ckpt, local_files_only=local_files_only
                )
                .eval()
                .to(DEVICE)
            )
            if self.enable_attention_maps:
                try:
                    self.model.set_attn_implementation("eager")
                    logging.info("Set model attention implementation to 'eager'.")
                except AttributeError:
                    logging.warning(
                        "Could not set attention implementation. Saliency masking might fail."
                    )
            else:
                logging.info(
                    "Attention maps disabled; using model default attention implementation."
                )
        logging.info("Model loaded successfully.")

    def _is_dinov3_checkpoint(self, model_ckpt: str) -> bool:
        ckpt_path = Path(model_ckpt)
        return ckpt_path.is_file() and ckpt_path.suffix == ".pth" and "dinov3" in ckpt_path.name

    def _load_dinov3_transformers_model(self, model_ckpt: str):
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))
        config = self._build_dinov3_transformers_config(state_dict)
        logging.info(
            "DINOv3 Transformers config: hidden_size=%s, layers=%s, heads=%s, intermediate_size=%s",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.intermediate_size,
        )
        model = DINOv3ViTModel(config)
        converted = self._convert_meta_dinov3_state_dict(state_dict)
        load_result = model.load_state_dict(converted, strict=False)
        allowed_missing = {
            f"layer.{idx}.attention.k_proj.bias"
            for idx in range(config.num_hidden_layers)
        }
        unexpected_missing = set(load_result.missing_keys) - allowed_missing
        if unexpected_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Failed to load local DINOv3 checkpoint. "
                f"Missing keys: {sorted(unexpected_missing)}; "
                f"Unexpected keys: {load_result.unexpected_keys}"
            )
        processor = DINOv3ViTImageProcessorFast()
        return processor, model.eval().to(DEVICE)

    @staticmethod
    def _build_dinov3_transformers_config(state_dict: dict) -> DINOv3ViTConfig:
        hidden_size = state_dict["cls_token"].shape[-1]
        patch_weight = state_dict["patch_embed.proj.weight"]
        intermediate_size = state_dict["blocks.0.mlp.fc1.weight"].shape[0]
        num_hidden_layers = 1 + max(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("blocks.") and key.endswith(".norm1.weight")
        )
        num_register_tokens = state_dict.get(
            "storage_tokens", torch.empty(1, 0, hidden_size)
        ).shape[1]
        patch_size = patch_weight.shape[-1]
        return DINOv3ViTConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=hidden_size // 64,
            num_register_tokens=num_register_tokens,
            layerscale_value=1.0e-05,
            patch_size=patch_size,
            image_size=512,
            attn_implementation="eager",
        )

    def _load_dinov3_native_model(self, model_ckpt: str):
        ckpt_name = Path(model_ckpt).name
        model_name = "_".join(ckpt_name.split("_")[:2])
        repo_candidates = [
            os.environ.get("DINOV3_REPO"),
            "/home/xrd/prj/UniADC-main",
            str(Path(model_ckpt).resolve().parent),
        ]
        repo_root = next(
            (
                Path(path)
                for path in repo_candidates
                if path and (Path(path) / "dinov3").is_dir()
            ),
            None,
        )
        if repo_root is None:
            raise FileNotFoundError(
                "Could not find a local DINOv3 python package. Set DINOV3_REPO to a "
                "directory containing the dinov3/ package."
            )
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from dinov3.hub.backbones import MODELS

        if model_name not in MODELS:
            raise ValueError(
                f"Unsupported DINOv3 checkpoint architecture '{model_name}'. "
                f"Available architectures: {sorted(MODELS)}"
            )

        model = MODELS[model_name](pretrained=False)
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _convert_meta_dinov3_state_dict(state_dict: dict) -> dict:
        """Converts Meta DINOv3 ViT-B/16 checkpoint keys to Transformers keys."""
        converted = {}

        for key, value in state_dict.items():
            new_key = None
            if key == "cls_token":
                new_key = "embeddings.cls_token"
            elif key == "mask_token":
                new_key = "embeddings.mask_token"
                value = value.unsqueeze(1)
            elif key == "storage_tokens":
                new_key = "embeddings.register_tokens"
            elif key.startswith("patch_embed.proj."):
                new_key = key.replace(
                    "patch_embed.proj.", "embeddings.patch_embeddings."
                )
            elif key.startswith("norm."):
                new_key = key
            elif ".attn.qkv." in key or key == "rope_embed.periods":
                continue
            elif ".attn.proj." in key:
                new_key = key.replace("blocks.", "layer.").replace(
                    ".attn.proj.", ".attention.o_proj."
                )
            elif ".ls1.gamma" in key:
                new_key = key.replace("blocks.", "layer.").replace(
                    ".ls1.gamma", ".layer_scale1.lambda1"
                )
            elif ".ls2.gamma" in key:
                new_key = key.replace("blocks.", "layer.").replace(
                    ".ls2.gamma", ".layer_scale2.lambda1"
                )
            elif ".mlp.fc1." in key:
                new_key = key.replace("blocks.", "layer.").replace(
                    ".mlp.fc1.", ".mlp.up_proj."
                )
            elif ".mlp.fc2." in key:
                new_key = key.replace("blocks.", "layer.").replace(
                    ".mlp.fc2.", ".mlp.down_proj."
                )
            elif ".norm1." in key or ".norm2." in key:
                new_key = key.replace("blocks.", "layer.")

            if new_key:
                converted[new_key] = value

        for key, value in state_dict.items():
            if key.endswith(".attn.qkv.weight"):
                base = key.replace("blocks.", "layer.").replace(
                    ".attn.qkv.weight", ".attention"
                )
                query, key_proj, value_proj = value.chunk(3, dim=0)
                converted[f"{base}.q_proj.weight"] = query
                converted[f"{base}.k_proj.weight"] = key_proj
                converted[f"{base}.v_proj.weight"] = value_proj
            elif key.endswith(".attn.qkv.bias"):
                base = key.replace("blocks.", "layer.").replace(
                    ".attn.qkv.bias", ".attention"
                )
                query, _, value_proj = value.chunk(3, dim=0)
                converted[f"{base}.q_proj.bias"] = query
                converted[f"{base}.v_proj.bias"] = value_proj

        return converted

    def _apply_clahe(self, pil_imgs: list) -> list:
        """Applies CLAHE to a list of PIL images."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_imgs = []
        for img in pil_imgs:
            img_np = np.array(img)
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_, a, b = cv2.split(img_lab)
            l_clahe = clahe.apply(l_)
            img_lab_clahe = cv2.merge((l_clahe, a, b))
            img_rgb_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
            processed_imgs.append(Image.fromarray(img_rgb_clahe))
        return processed_imgs

    def _spatial_from_seq(
        self,
        seq_tokens: torch.Tensor,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        """Converts a sequence of tokens to a spatial (grid) format."""
        B, N, C = seq_tokens.shape
        tokens = seq_tokens[:, drop_front : drop_front + n_expected, :]
        return tokens.reshape(B, h_p, w_p, C)

    def _get_saliency_mask(
        self,
        attentions: tuple,
        dino_saliency_layer: int,
        num_reg: int,
        drop_front: int,
        n_expected: int,
        batch_size: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        """Extracts the DINO saliency mask from attention weights."""
        if dino_saliency_layer < 0:
            dino_saliency_layer = len(attentions) + dino_saliency_layer

        if dino_saliency_layer >= len(attentions):
            logging.warning(
                f"DINO saliency layer {dino_saliency_layer} is out of bounds (0-{len(attentions)-1}). Defaulting to 0."
            )
            dino_saliency_layer = 0

        attn_map = attentions[dino_saliency_layer]
        if num_reg > 0:
            reg_attn_to_patches = attn_map[
                :, :, 1:drop_front, drop_front : drop_front + n_expected
            ]
            saliency_mask = reg_attn_to_patches.mean(dim=(1, 2))
        else:
            logging.info("No register tokens found. Using CLS token for saliency mask.")
            cls_attn_to_patches = attn_map[
                :, :, 0, drop_front : drop_front + n_expected
            ]
            saliency_mask = cls_attn_to_patches.mean(dim=1)

        return saliency_mask.reshape(batch_size, h_p, w_p).cpu().numpy()

    def _aggregate_layers(
        self,
        hidden_states: tuple,
        layers: list,
        grouped_layers: list,
        agg_method: str,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        """Aggregates features from specified layers."""

        _spatial_converter = lambda x: self._spatial_from_seq(
            x, drop_front, n_expected, h_p, w_p
        )

        if agg_method == "group":
            if not grouped_layers:
                raise ValueError(
                    "Grouped layers must be provided for 'group' aggregation."
                )

            all_layer_indices = sorted(
                list(set(idx for group in grouped_layers for idx in group))
            )
            layer_tensors = {
                li: _spatial_converter(hidden_states[li]) for li in all_layer_indices
            }
            fused_groups = [
                torch.stack([layer_tensors[li] for li in group], dim=0).mean(dim=0)
                for group in grouped_layers
            ]
            fused = torch.cat(fused_groups, dim=-1)

        else:
            feats = [_spatial_converter(hidden_states[li]) for li in layers]
            if agg_method == "concat":
                fused = torch.cat(feats, dim=-1)
            elif agg_method == "mean":
                fused = torch.stack(feats, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: '{agg_method}'")

        return fused.cpu().numpy()

    def _preprocess_dinov3_native_images(
        self,
        pil_imgs: list,
        res: int,
        docrop: bool,
    ) -> torch.Tensor:
        if docrop:
            resize_res = int(res / 0.875)
            resize_size = [resize_res, resize_res]
        else:
            resize_size = [res, res]

        tensors = []
        for img in pil_imgs:
            img = img.convert("RGB")
            img = TF.resize(img, resize_size, interpolation=InterpolationMode.BICUBIC)
            if docrop:
                img = TF.center_crop(img, [res, res])
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0).to(DEVICE)

    def _canonical_dinov3_native_layer(self, layer: int, total_layers: int) -> int:
        if layer < 0:
            layer = total_layers + layer
        if layer < 0 or layer >= total_layers:
            raise ValueError(
                f"DINOv3 layer index {layer} is out of bounds for {total_layers} blocks."
            )
        return layer

    def _spatial_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        n_expected = h_p * w_p
        if N != n_expected:
            raise ValueError(
                f"Expected {n_expected} patch tokens for grid {h_p}x{w_p}, got {N}."
            )
        return patch_tokens.reshape(B, h_p, w_p, C)

    def _aggregate_dinov3_native_layers(
        self,
        patch_tokens_by_layer: dict,
        layers: list,
        grouped_layers: list,
        agg_method: str,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        def to_spatial(layer: int) -> torch.Tensor:
            return self._spatial_from_patch_tokens(
                patch_tokens_by_layer[layer],
                h_p,
                w_p,
            )

        if agg_method == "group":
            if not grouped_layers:
                raise ValueError(
                    "Grouped layers must be provided for 'group' aggregation."
                )
            fused_groups = [
                torch.stack([to_spatial(li) for li in group], dim=0).mean(dim=0)
                for group in grouped_layers
            ]
            fused = torch.cat(fused_groups, dim=-1)
        else:
            feats = [to_spatial(li) for li in layers]
            if agg_method == "concat":
                fused = torch.cat(feats, dim=-1)
            elif agg_method == "mean":
                fused = torch.stack(feats, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: '{agg_method}'")
        return fused.cpu().numpy()

    def _extract_tokens_dinov3_native(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list = [],
        docrop: bool = False,
        use_clahe: bool = False,
    ):
        if use_clahe:
            pil_imgs = self._apply_clahe(pil_imgs)

        inputs = self._preprocess_dinov3_native_images(pil_imgs, res, docrop)
        patch_size = getattr(self.model, "patch_size", 16)
        h_p, w_p = inputs.shape[-2] // patch_size, inputs.shape[-1] // patch_size
        total_layers = len(self.model.blocks)
        canonical_layers = [
            self._canonical_dinov3_native_layer(layer, total_layers)
            for layer in layers
        ]
        canonical_groups = [
            [
                self._canonical_dinov3_native_layer(layer, total_layers)
                for layer in group
            ]
            for group in grouped_layers
        ]
        requested_layers = sorted(
            set(canonical_layers + [layer for group in canonical_groups for layer in group])
        )
        outputs = self.model.get_intermediate_layers(
            inputs,
            n=requested_layers,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        patch_tokens_by_layer = dict(zip(requested_layers, outputs))
        fused_tokens = self._aggregate_dinov3_native_layers(
            patch_tokens_by_layer,
            canonical_layers,
            canonical_groups,
            agg_method,
            h_p,
            w_p,
        )
        return fused_tokens, (h_p, w_p), None

    @torch.no_grad()
    def extract_tokens(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list = [],
        docrop: bool = False,
        use_clahe: bool = False,
        dino_saliency_layer: int = 0,
    ):
        """
        Extracts, aggregates features, and computes saliency from a batch of images.

        Returns:
            - fused_tokens (np.ndarray): The aggregated patch features.
            - grid_size (tuple): The (height, width) of the patch grid.
            - saliency_mask (np.ndarray): The DINO saliency mask.
        """

        if self.backend == "dinov3_native":
            return self._extract_tokens_dinov3_native(
                pil_imgs,
                res,
                layers,
                agg_method,
                grouped_layers=grouped_layers,
                docrop=docrop,
                use_clahe=use_clahe,
            )

        # 1. Preprocessing
        if use_clahe:
            pil_imgs = self._apply_clahe(pil_imgs)

        if docrop:
            resize_res = int(res / 0.875)
            size = {"height": resize_res, "width": resize_res}
            crop_size = {"height": res, "width": res}
        else:
            size = {"height": res, "width": res}
            crop_size = {"height": res, "width": res}

        inputs = self.processor(
            images=pil_imgs,
            return_tensors="pt",
            do_resize=True,
            size=size,
            do_center_crop=docrop,
            crop_size=crop_size,
        ).to(DEVICE)

        # 2. Model Inference
        output_attentions = (
            self.enable_attention_maps or self.backend == "dinov3_transformers"
        )
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions if output_attentions else None

        # 3. Setup Parameters
        cfg = self.model.config
        ps = cfg.patch_size
        num_reg = getattr(cfg, "num_register_tokens", 0)
        drop_front = 1 + num_reg  # CLS token + register tokens
        h_p, w_p = res // ps, res // ps
        n_expected = h_p * w_p
        batch_size = inputs.pixel_values.shape[0]

        # 4. Saliency Mask Extraction
        saliency_mask = None
        if self.enable_attention_maps:
            if attentions is None:
                raise ValueError(
                    "Attention weights are None. Model may be using Flash Attention. "
                    "Check transformers version or model compatibility."
                )
            saliency_mask = self._get_saliency_mask(
                attentions,
                dino_saliency_layer,
                num_reg,
                drop_front,
                n_expected,
                batch_size,
                h_p,
                w_p,
            )

        # 5. Feature Aggregation
        fused_tokens = self._aggregate_layers(
            hidden_states,
            layers,
            grouped_layers,
            agg_method,
            drop_front,
            n_expected,
            h_p,
            w_p,
        )

        return fused_tokens, (h_p, w_p), saliency_mask
