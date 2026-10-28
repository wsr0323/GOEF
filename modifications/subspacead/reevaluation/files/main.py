import os
import math
import logging
from pathlib import Path

import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.decomposition import PCA

from src.subspacead.config import get_args, parse_layer_indices, parse_grouped_layers
from src.subspacead.utils.common import (
    setup_logging,
    save_config,
    min_max_norm,
)
from src.subspacead.data.datasets import get_dataset_handler
from src.subspacead.core.extractor import FeatureExtractor
from src.subspacead.core.pca import PCAModel, KernelPCAModel
from src.subspacead.post_process.scoring import (
    calculate_anomaly_scores,
    post_process_map,
)
from src.subspacead.utils.viz import save_visualization, save_overlay_for_intro
from src.subspacead.post_process.specular import (
    specular_mask_torch,
    filter_specular_anomalies,
)
from src.subspacead.core.patching import process_image_patched, get_patch_coords
from src.subspacead.data.transforms import get_augmentation_transform
from unified_evaluation import (
    InferenceBenchmark,
    compute_metrics,
    format_result,
    make_result_row,
    write_results_csv,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


def topk_mean(arr, frac=0.01):
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def main():
    args = get_args()
    # Benchmark runs use a single, fixed seed across all projects.
    args.seed = 42
    run_name = f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}_res{args.image_res}_docrop{int(args.docrop)}"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        print("No seed specified; aborting for reproducibility.")
        return
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_specular_filter:
        run_name += "_spec-filt"
    if args.bg_mask_method:
        run_name += f"_mask-{args.bg_mask_method}_thr-{args.mask_threshold_method}"
        if args.mask_threshold_method == "percentile":
            run_name += f"{args.percentile_threshold}"
        if args.bg_mask_method == "dino_saliency":
            run_name += f"_L{args.dino_saliency_layer}"
    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"
    run_name += f"_model-{args.model_ckpt.split('/')[-1]}"
    run_name += (
        f"pca_ev{args.pca_ev}" if args.pca_ev is not None else f"_pca_dim{args.pca_dim}"
    )
    run_name += f"_i-score{args.img_score_agg}"

    # Add k-shot and augmentation info to run name
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            # Create a short string for augs, e.g., "hrc"
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"

    run_name += f"_seed{args.seed}"

    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

    # Augmentations
    aug_transform = None
    if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
        aug_transform = get_augmentation_transform(args.aug_list, args.image_res)
        if not aug_transform.transforms:
            logging.warning(
                "Augmentation specified but no valid transforms were created. Disabling augmentations."
            )
            aug_transform = None

    # Parse layer args
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    # Init model
    extractor = FeatureExtractor(args.model_ckpt, args.local_files_only)

    # Get dataset categories
    if args.categories:
        categories = args.categories
    else:
        categories = sorted(
            [
                f.name
                for f in Path(args.dataset_path).iterdir()
                if f.is_dir() and f.name != "split_csv"
            ]
        )

    # Main loop
    unified_rows = []
    for category in categories:
        logging.info(f"--- Processing Category: {category} ---")

        if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
            aug_transform = get_augmentation_transform(args.aug_list, args.image_res)

        else:
            aug_transform = None
        if category in args.no_aug_categories:
            logging.warning(f"Disabling augmentation for {category} category")
            aug_transform = None
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)
        train_paths = handler.get_train_paths()
        test_paths = handler.get_test_paths()

        if args.debug_limit is not None:
            logging.warning(
                f"--- DEBUG MODE: Limiting test set to {args.debug_limit} images ---"
            )
            if test_paths:
                test_paths = test_paths[: args.debug_limit]

        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        if args.batched_zero_shot:
            # Batched 0-shot train=test
            logging.info(
                f"--- Batched 0-Shot Mode: Fitting PCA on {len(test_paths)} test images ---"
            )
            train_paths = test_paths.copy()

        # K-shot sampling
        if args.k_shot is not None:
            if args.k_shot > len(train_paths):
                logging.warning(
                    f"Requested k_shot={args.k_shot} but only {len(train_paths)} training images available. Using all {len(train_paths)}."
                )
            else:
                logging.info(
                    f"--- K-SHOT: Randomly sampling {args.k_shot} training images ---"
                )
                random.shuffle(train_paths)
                train_paths = (
                    train_paths[: args.k_shot]
                    if args.k_shot <= len(train_paths)
                    else train_paths
                )
                for i, path in enumerate(train_paths):
                    logging.info(
                        f"  K-Shot image {i + 1}/{args.k_shot}: {Path(path).name}"
                    )

        # 1. Fit PCA Model
        if args.patch_size:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with --patch_size. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with patch_size.")

            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_patch],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            tokens_per_patch = h_p * w_p

            # Calculate total number of patches and tokens (with augmentations)
            total_patches = 0
            num_batches = 0
            # This multiplier accounts for the original image + N augmented images
            num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1

            for path in train_paths:
                img = Image.open(path).convert("RGB")
                patch_coords = get_patch_coords(
                    img.height, img.width, args.patch_size, args.patch_overlap
                )
                total_patches += len(patch_coords) * num_aug_multiplier
                num_batches += (
                    math.ceil(len(patch_coords) / args.batch_size) * num_aug_multiplier
                )
            total_tokens = total_patches * tokens_per_patch

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, "
                f"Base train patches: {total_patches // num_aug_multiplier}, "
                f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
            )

            def feature_generator_patched():
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")

                    # Create a list of images to process: original + augmentations
                    images_to_process = [pil_img]
                    if aug_transform:
                        for _ in range(args.aug_count):
                            images_to_process.append(aug_transform(pil_img))

                    # Process each image (original + augmented)
                    for img in images_to_process:
                        patch_coords = get_patch_coords(
                            img.height,
                            img.width,
                            args.patch_size,
                            args.patch_overlap,
                        )
                        for i in range(0, len(patch_coords), args.batch_size):
                            coord_batch = patch_coords[i : i + args.batch_size]
                            patch_batch = [img.crop(c) for c in coord_batch]
                            (
                                tokens_batch,
                                _,
                                saliency_masks_batch,
                            ) = extractor.extract_tokens(
                                patch_batch,
                                args.image_res,
                                layers,
                                args.agg_method,
                                grouped_layers,
                                args.docrop,
                                use_clahe=args.use_clahe,
                                dino_saliency_layer=args.dino_saliency_layer,
                            )
                            tokens_flat = tokens_batch.reshape(-1, feature_dim)

                            if args.bg_mask_method == "dino_saliency":
                                masks_flat = saliency_masks_batch.reshape(-1)
                                try:
                                    if args.mask_threshold_method == "percentile":
                                        threshold = np.percentile(
                                            masks_flat, args.percentile_threshold * 100
                                        )
                                        foreground_tokens = tokens_flat[
                                            masks_flat >= threshold
                                        ]
                                    else:
                                        norm_mask = cv2.normalize(
                                            masks_flat,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        foreground_tokens = tokens_flat[
                                            binary_mask.flatten() > 0
                                        ]

                                    if foreground_tokens.shape[0] > 0:
                                        yield foreground_tokens
                                    else:
                                        logging.warning(
                                            "No foreground patch tokens found. Yielding all tokens."
                                        )
                                        yield tokens_flat
                                except Exception as e:
                                    logging.warning(
                                        f"Masking failed: {e}. Yielding all tokens."
                                    )
                                    yield tokens_flat
                            else:
                                yield tokens_flat

            feature_generator = feature_generator_patched

        else:
            # PCA without patching
            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_img],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1
            total_train_images = len(train_paths) * num_aug_multiplier
            total_tokens = total_train_images * h_p * w_p

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
                f"Base train images: {len(train_paths)}, "
                f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
            )

            def feature_generator_full():
                all_imgs_to_process = []
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")
                    all_imgs_to_process.append(pil_img)
                    if aug_transform:
                        for _ in range(args.aug_count):
                            all_imgs_to_process.append(aug_transform(pil_img))

                # Now process all_imgs_to_process in batches
                for i in range(0, len(all_imgs_to_process), args.batch_size):
                    img_batch = all_imgs_to_process[i : i + args.batch_size]
                    (
                        tokens_batch,
                        _,
                        saliency_masks_batch,
                    ) = extractor.extract_tokens(
                        img_batch,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                    )
                    tokens_flat = tokens_batch.reshape(-1, feature_dim)

                    # Train masking logic
                    if args.bg_mask_method == "dino_saliency":
                        masks_flat = saliency_masks_batch.reshape(-1)
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(
                                    masks_flat, args.percentile_threshold * 100
                                )
                                foreground_tokens = tokens_flat[masks_flat >= threshold]
                            else:
                                norm_mask = cv2.normalize(
                                    masks_flat,
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U,
                                )
                                _, binary_mask = cv2.threshold(
                                    norm_mask,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                foreground_tokens = tokens_flat[
                                    binary_mask.flatten() > 0
                                ]

                            if foreground_tokens.shape[0] > 0:
                                yield foreground_tokens
                            else:
                                logging.warning(
                                    "No foreground tokens found. Yielding all tokens."
                                )
                                yield tokens_flat
                        except Exception as e:
                            logging.warning(
                                f"Masking failed: {e}. Yielding all tokens."
                            )
                            yield tokens_flat
                    else:
                        yield tokens_flat

            num_batches = math.ceil(total_train_images / args.batch_size)
            feature_generator = feature_generator_full

        if args.use_kernel_pca:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with Kernel PCA. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with use_kernel_pca.")

            logging.info("Collecting all features for Kernel PCA...")
            all_train_tokens = np.concatenate(
                list(
                    tqdm(
                        feature_generator(),
                        desc="Feature Collection",
                        total=num_batches,
                    )
                )
            )
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
            pca_params = pca_model.fit(
                feature_generator,
                feature_dim,
                total_tokens,
                num_batches,
            )

        # 2. Evaluate on Test Set using the unified protocol.
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        img_true = []
        img_pred_auroc = []
        pro_gt_masks = []
        pro_anomaly_maps = []
        vis_saved_count = 0
        all_inference_times = []
        benchmark = InferenceBenchmark(DEVICE)

        logging.info("Number of test images: {}".format(len(test_paths)))
        test_iter = tqdm(test_paths, desc=f"Testing {category}")
        for i in range(0, len(test_paths), args.batch_size):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = [
                "good" not in str(p) and "Normal" not in str(p) for p in path_batch
            ]
            benchmark.start_batch()

            final_anomaly_maps_for_batch = []
            saliency_maps_for_viz_batch = []

            if args.patch_size:
                (
                    anomaly_maps_batch,
                    saliency_maps_batch,
                ) = process_image_patched(
                    pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
                )

                saliency_maps_for_viz_batch = saliency_maps_batch

                for j, anomaly_map_pre_specular in enumerate(anomaly_maps_batch):
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_pre_specular.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(
                                anomaly_map_pre_specular, conf_map
                            )
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            else:
                # Step 1: Feature Extraction
                (
                    tokens,
                    (h_p, w_p),
                    saliency_masks_batch,
                ) = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    use_clahe=args.use_clahe,
                    dino_saliency_layer=args.dino_saliency_layer,
                )
                b, _, _, c = tokens.shape
                tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

                # Step 2: Anomaly Scoring
                scores = calculate_anomaly_scores(
                    tokens_reshaped,
                    pca_params,
                    args.score_method,
                    args.drop_k,
                )
                anomaly_maps = scores.reshape(b, h_p, w_p)

                # Step 3: Masking Strategy
                mask_for_viz = None
                background_mask = np.zeros_like(anomaly_maps, dtype=bool)

                if args.bg_mask_method == "dino_saliency":
                    mask_for_viz = saliency_masks_batch
                    for j in range(b):
                        saliency_map = saliency_masks_batch[j]
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(
                                    saliency_map, args.percentile_threshold * 100
                                )
                                background_mask[j] = saliency_map < threshold
                            else:  # otsu
                                norm_mask = cv2.normalize(
                                    saliency_map,
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U,
                                )
                                _, binary_mask = cv2.threshold(
                                    norm_mask,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                background_mask[j] = binary_mask == 0
                        except Exception as e:
                            logging.warning(
                                f"Saliency mask failed for test image {j}: {e}. Skipping mask."
                            )

                elif args.bg_mask_method == "pca_normality":
                    threshold = 10.0
                    kernel_size = 3
                    border = 0.2
                    grid_size = (h_p, w_p)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_for_viz = np.zeros_like(anomaly_maps)

                    for j in range(b):
                        img_features = tokens[j].reshape(-1, c)
                        try:
                            pca = PCA(n_components=1, svd_solver="randomized")
                            first_pc = pca.fit_transform(
                                img_features.astype(np.float32)
                            )
                            mask = first_pc > threshold
                            mask_2d = mask.reshape(grid_size)

                            h_start, h_end = int(grid_size[0] * border), int(
                                grid_size[0] * (1 - border)
                            )
                            w_start, w_end = int(grid_size[1] * border), int(
                                grid_size[1] * (1 - border)
                            )
                            m = mask_2d[h_start:h_end, w_start:w_end]

                            if m.sum() <= m.size * 0.35:
                                mask = -first_pc > threshold
                                mask_2d = mask.reshape(grid_size)

                            mask_processed = cv2.dilate(
                                mask_2d.astype(np.uint8), kernel
                            ).astype(bool)
                            mask_processed = cv2.morphologyEx(
                                mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                            ).astype(bool)

                            background_mask[j] = ~mask_processed
                            mask_for_viz[j] = mask_processed.astype(np.float32)
                        except Exception as e:
                            logging.warning(
                                f"PCA mask failed for test image {j}: {e}. Skipping mask."
                            )

                anomaly_maps[background_mask] = 0.0

                saliency_maps_for_viz_batch = mask_for_viz

                for j in range(anomaly_maps.shape[0]):
                    pil_img = pil_imgs[j]
                    anomaly_map_pre_specular = post_process_map(
                        anomaly_maps[j], args.image_res
                    )
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_final.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(anomaly_map_final, conf_map)
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            batch_img_scores = []
            for anomaly_map_final in final_anomaly_maps_for_batch:
                if args.img_score_agg == "max":
                    img_score = np.max(anomaly_map_final)
                elif args.img_score_agg == "p99":
                    img_score = np.percentile(anomaly_map_final, 99)
                elif args.img_score_agg == "mtop5":
                    img_score = np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                elif args.img_score_agg == "mtop1p":
                    img_score = topk_mean(anomaly_map_final, frac=0.01)
                else:
                    img_score = np.mean(anomaly_map_final)
                batch_img_scores.append(float(img_score))

            all_inference_times.append(benchmark.stop_batch(len(path_batch)))

            for j, anomaly_map_final in enumerate(final_anomaly_maps_for_batch):
                is_anomaly = is_anomaly_batch[j]
                path = path_batch[j]
                pil_img = pil_imgs[j]
                img_score = batch_img_scores[j]

                img_true.append(1 if is_anomaly else 0)
                img_pred_auroc.append(float(img_score))
                anomaly_map_normalized = min_max_norm(anomaly_map_final)
                H, W = anomaly_map_normalized.shape
                gt_path_str = handler.get_ground_truth_path(path)
                if not gt_path_str or not os.path.exists(gt_path_str):
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    gt_mask_pil = Image.open(gt_path_str).convert("L")
                    if args.docrop:
                        resize_res = int(args.image_res / 0.875)
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask_pil = TF.center_crop(
                            gt_mask_pil, (args.image_res, args.image_res)
                        )
                    gt_mask_pil = TF.resize(
                        gt_mask_pil,
                        (H, W),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                pro_gt_masks.append(gt_mask)
                pro_anomaly_maps.append(anomaly_map_final.astype(np.float32))

                if is_anomaly:
                    if args.save_intro_overlays:
                        vis_img = pil_img
                        save_overlay_for_intro(
                            path,
                            vis_img,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                        )
                    if vis_saved_count < args.vis_count:
                        vis_img = pil_img
                        if args.docrop and not args.patch_size:
                            resize_res = int(args.image_res / 0.875)
                            vis_img = TF.resize(
                                vis_img,
                                (resize_res, resize_res),
                                interpolation=TF.InterpolationMode.BICUBIC,
                            )
                            vis_img = TF.center_crop(
                                vis_img, (args.image_res, args.image_res)
                            )
                        saliency_map_for_viz = None
                        raw_mask_map = None
                        if saliency_maps_for_viz_batch is not None:
                            raw_mask_map = saliency_maps_for_viz_batch[j]

                        if raw_mask_map is not None:
                            try:
                                if args.bg_mask_method == "pca_normality":
                                    binary_mask = raw_mask_map

                                elif args.bg_mask_method == "dino_saliency":
                                    if args.mask_threshold_method == "percentile":
                                        threshold_val = np.percentile(
                                            raw_mask_map,
                                            args.percentile_threshold * 100,
                                        )
                                        binary_mask = (
                                            raw_mask_map >= threshold_val
                                        ).astype(np.float32)
                                    else:  # otsu
                                        norm_mask = cv2.normalize(
                                            raw_mask_map,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask_u8 = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        binary_mask = (binary_mask_u8 > 0).astype(
                                            np.float32
                                        )
                                saliency_map_for_viz = post_process_map(
                                    binary_mask,
                                    anomaly_map_normalized.shape,
                                    blur=False,
                                )
                            except Exception as e:
                                logging.warning(
                                    f"Saliency mask processing failed for visualization: {e}."
                                )

                        save_visualization(
                            path,
                            vis_img,
                            gt_mask,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                            vis_saved_count,
                            saliency_mask=saliency_map_for_viz,
                        )
                        vis_saved_count += 1

            test_iter.update(len(path_batch))
        benchmark_stats = benchmark.result()
        if all_inference_times:
            logging.info(f"--- Timing Results for {category} ---")
            logging.info(f"Total test images: {benchmark_stats['Images']}")
            logging.info(
                f"Batch size: {args.batch_size} (Processed {len(all_inference_times)} batches)"
            )
            logging.info(f"Total inference time: {benchmark_stats['Inference Total (s)']:.4f} s")
            logging.info(f"Avg. time per image: {benchmark_stats['Inference (ms/image)']:.3f} ms")
            logging.info(f"Images per second (FPS): {benchmark_stats['Throughput (images/s)']:.2f}")
            logging.info(
                f"GPU peak: {benchmark_stats['GPU Peak Allocated (MiB)']:.1f} MiB allocated, "
                f"{benchmark_stats['GPU Peak Reserved (MiB)']:.1f} MiB reserved"
            )

            # Report batch stats
            times_arr = np.array(all_inference_times)
            if len(times_arr) > 1:
                times_arr_stats = times_arr[1:]
                logging.info(
                    f"Avg. time per batch (excl. 1st): {np.mean(times_arr_stats):.6f} s"
                )
                logging.info(
                    f"Median time per batch (excl. 1st): {np.median(times_arr_stats):.6f} s"
                )
            else:
                logging.info(f"Avg. time per batch: {np.mean(times_arr):.6f} s")
        unified_metrics = compute_metrics(
            img_true,
            img_pred_auroc,
            pro_gt_masks,
            pro_anomaly_maps,
        )
        unified_row = make_result_row(category, unified_metrics, benchmark_stats)
        unified_rows.append(unified_row)
        logging.info(format_result(unified_row))

    unified_path = write_results_csv(
        os.path.join(args.outdir, "unified_results.csv"), unified_rows
    )
    logging.info(f"Unified results saved to {unified_path}")


if __name__ == "__main__":
    main()
