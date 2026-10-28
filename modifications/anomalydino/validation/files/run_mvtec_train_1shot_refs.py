import argparse
import csv
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import yaml
from tqdm import tqdm, trange


def parse_args():
    parser = ArgumentParser(
        description=(
            "Run MVTec one-shot anomaly detection once for every train/good "
            "image, using each image as the only reference sample."
        )
    )
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--preprocess", type=str, default="informed")
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument(
        "--faiss_on_cpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run FAISS similarity search on CPU instead of GPU.",
    )
    parser.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_examples", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction)

    # 将 eval_segm 的默认值改为了 True
    parser.add_argument("--eval_segm", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument(
        "--delete_tiff_files",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete TIFF anomaly maps after segmentation evaluation.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup_iters", type=int, default=0)
    object_group = parser.add_mutually_exclusive_group()
    object_group.add_argument(
        "--object",
        default=None,
        help="Optional single MVTec object to run, e.g. --object bottle.",
    )
    object_group.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Optional subset of MVTec objects to run, e.g. --objects bottle cable.",
    )
    parser.add_argument(
        "--start_ref_index",
        type=int,
        default=0,
        help="Skip train/good references before this sorted index for each object.",
    )
    parser.add_argument(
        "--max_refs_per_object",
        type=int,
        default=None,
        help="Optional cap for debugging, e.g. --max_refs_per_object 3.",
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def safe_ref_dir_name(ref_index, ref_name):
    return f"ref_{ref_index:04d}_{Path(ref_name).stem}"


def list_reference_images(data_root, object_name, start_ref_index, max_refs_per_object):
    ref_dir = Path(data_root) / object_name / "train" / "good"
    ref_names = sorted(path.name for path in ref_dir.iterdir() if path.is_file())
    ref_names = ref_names[start_ref_index:]
    if max_refs_per_object is not None:
        ref_names = ref_names[:max_refs_per_object]
    return ref_names


def evaluate_object_run(
        object_name,
        data_root,
        anomaly_maps_dir,
        output_dir,
        eval_clf=True,
        eval_segm=True,  # 这里也同步改为了 True
        delete_tiff_files=True,
):
    from src.post_eval import eval_classification, eval_segmentation, parse_dataset_files

    gt_filenames, prediction_filenames = parse_dataset_files(
        object_name=object_name,
        dataset_base_dir=data_root,
        anomaly_maps_dir=anomaly_maps_dir,
        dataset="MVTec",
    )

    metrics = {}
    if eval_clf:
        auroc_clf, ap_clf, f1_clf = eval_classification(gt_filenames, prediction_filenames)
        metrics["classification_AUROC"] = auroc_clf
        metrics["classification_AP"] = ap_clf
        metrics["classification_F1"] = f1_clf

    if eval_segm:
        au_pro, auroc_px, f1_px = eval_segmentation(
            gt_filenames,
            prediction_filenames,
            pro_integration_limit=0.3,
            delete_tiff_files=delete_tiff_files,
        )
        metrics["seg_AUPRO"] = au_pro
        metrics["seg_AUROC"] = auroc_px
        metrics["seg_F1"] = f1_px

    metrics_path = Path(output_dir) / "metrics.json"
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=4)
    return metrics


def append_summary_row(summary_file, row):
    exists = summary_file.exists()
    with summary_file.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "object",
                "Total_Train_Images",  # 新增：记录该类别总遍历张数
                "ref_index",
                "ref_sample",
                "run_dir",
                "Image AUROC",  # 重命名
                "Image AUPR",  # 重命名
                "classification_F1",
                "Pixel AUPRO",  # 重命名
                "Pixel AUROC",  # 重命名
                "seg_F1",
                "mean_inference_time",
                "status",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    from src.backbones import get_model
    from src.detection import run_anomaly_detection
    from src.utils import get_dataset_info

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        "MVTec",
        args.preprocess,
        data_path=args.data_root,
    )
    selected_objects = [args.object] if args.object is not None else args.objects
    if selected_objects is not None:
        unknown_objects = sorted(set(selected_objects) - set(objects))
        if unknown_objects:
            raise ValueError(f"Unknown MVTec object(s): {unknown_objects}")
        objects = selected_objects

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, "cuda", smaller_edge_size=args.resolution)

    if not args.model_name.startswith("dinov2"):
        masking_default = {object_name: False for object_name in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")

    if args.output_root is None:
        output_root = (
                Path(f"results_MVTec/{args.model_name}_{args.resolution}")
                / f"train-all-1-shot_preprocess={args.preprocess}"
        )
        if args.tag is not None:
            output_root = Path(str(output_root) + f"_{args.tag}")
    else:
        output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "args.yaml").open("w") as file:
        yaml.dump(vars(args), file)
    with (output_root / "preprocess.yaml").open("w") as file:
        yaml.dump({"masking": masking_default, "rotation": rotation_default}, file)

    summary_file = output_root / "summary.csv"
    print("Results will be saved to", output_root)

    if args.faiss_on_cpu:
        print("Warning: Running similarity search on CPU. Consider faiss-gpu for faster inference.")

    for object_name in tqdm(objects, desc="MVTec objects"):
        ref_names = list_reference_images(
            args.data_root,
            object_name,
            args.start_ref_index,
            args.max_refs_per_object,
        )
        if not ref_names:
            print(f"Warning: no train/good references found for {object_name}.")
            continue

        if args.warmup_iters > 0:
            first_image = ref_names[0]
            first_image_path = Path(args.data_root) / object_name / "train" / "good" / first_image
            for _ in trange(args.warmup_iters, desc=f"CUDA warmup ({object_name})", leave=False):
                img_tensor, _ = model.prepare_image(str(first_image_path))
                model.extract_features(img_tensor)

        for local_ref_index, ref_name in enumerate(tqdm(ref_names, desc=object_name, leave=False)):
            ref_index = args.start_ref_index + local_ref_index
            run_dir = output_root / object_name / safe_ref_dir_name(ref_index, ref_name)
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                print(f"Skip existing run: {run_dir}")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "reference.yaml").open("w") as file:
                yaml.dump(
                    {
                        "object": object_name,
                        "ref_index": ref_index,
                        "ref_sample": ref_name,
                    },
                    file,
                )

            anomaly_scores, time_memorybank, time_inference = run_anomaly_detection(
                model,
                object_name,
                data_root=args.data_root,
                n_ref_samples=1,
                object_anomalies=object_anomalies,
                plots_dir=str(run_dir),
                save_examples=args.save_examples,
                knn_metric=args.knn_metric,
                knn_neighbors=args.k_neighbors,
                faiss_on_cpu=args.faiss_on_cpu,
                masking=masking_default[object_name],
                mask_ref_images=args.mask_ref_images,
                rotation=rotation_default[object_name],
                seed=0,
                ref_sample_names=[ref_name],
                save_patch_dists=args.eval_clf,
                save_tiffs=args.eval_segm,  # 此处会自动跟随上面的 True 生效
            )

            measurements_file = run_dir / "measurements.csv"
            with measurements_file.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Object", "Reference", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"]
                )
                for sample_name, anomaly_score in anomaly_scores.items():
                    writer.writerow(
                        [
                            object_name,
                            ref_name,
                            sample_name,
                            f"{anomaly_score:.5f}",
                            f"{time_memorybank:.5f}",
                            f"{time_inference[sample_name]:.5f}",
                        ]
                    )

            metrics = {}
            if args.eval_clf or args.eval_segm:
                metrics = evaluate_object_run(
                    object_name=object_name,
                    data_root=args.data_root,
                    anomaly_maps_dir=str(run_dir / "anomaly_maps" / "seed=0"),
                    output_dir=run_dir,
                    eval_clf=args.eval_clf,
                    eval_segm=args.eval_segm,
                    delete_tiff_files=args.delete_tiff_files,
                )

            inference_values = list(time_inference.values())
            mean_inference_time = sum(inference_values) / max(len(inference_values), 1)

            # 写入新的汇总表，并记录总张数
            append_summary_row(
                summary_file,
                {
                    "object": object_name,
                    "Total_Train_Images": len(ref_names),  # 该类别遍历的总张数
                    "ref_index": ref_index,
                    "ref_sample": ref_name,
                    "run_dir": str(run_dir),
                    "Image AUROC": metrics.get("classification_AUROC", ""),
                    "Image AUPR": metrics.get("classification_AP", ""),
                    "classification_F1": metrics.get("classification_F1", ""),
                    "Pixel AUPRO": metrics.get("seg_AUPRO", ""),
                    "Pixel AUROC": metrics.get("seg_AUROC", ""),
                    "seg_F1": metrics.get("seg_F1", ""),
                    "mean_inference_time": f"{mean_inference_time:.5f}",
                    "status": "done",
                },
            )

    print("Finished all requested MVTec train-image 1-shot runs.")


if __name__ == "__main__":
    main()