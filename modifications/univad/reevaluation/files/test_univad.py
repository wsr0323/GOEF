import argparse
import json
import logging
import os
import random
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math
from PIL import Image
from prefetch_generator import BackgroundGenerator
from UniVAD import UniVAD
from unified_evaluation import (
    InferenceBenchmark,
    compute_metrics,
    format_result,
    make_result_row,
    write_results_csv,
)

from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from datasets.mvtec_loco import MVTecLocoDataset
from datasets.brainmri import BrainMRIDataset
from datasets.his import HISDataset
from datasets.resc import RESCDataset
from datasets.liverct import LiverCTDataset
from datasets.chestxray import ChestXrayDataset
from datasets.oct17 import OCT17Dataset


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def resize_tokens(x):
    B, N, C = x.shape
    x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
    return x


def get_grounding_mask_path(image_path, dataset_root, mask_root):
    relative_path = os.path.relpath(image_path, dataset_root)
    relative_path = os.path.splitext(relative_path)[0]
    return os.path.join(mask_root, relative_path, "grounding_mask.png")


def cal_score(obj):
    table = []
    gt_px = []
    pr_px = []
    gt_sp = []
    pr_sp = []

    table.append(obj)
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)

    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())

    table.append(str(np.round(auroc_sp * 100, decimals=1)))
    table.append(str(np.round(auroc_px * 100, decimals=1)))

    table_ls.append(table)
    auroc_sp_ls.append(auroc_sp)
    auroc_px_ls.append(auroc_px)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/mvtec",
        help="path to test dataset",
    )
    parser.add_argument(
        "--save_path", type=str, default=f"./results/", help="path to save results"
    )
    parser.add_argument(
        "--round", type=int, default=3, help="round"
    )
    parser.add_argument("--class_name", type=str, default="None", help="device")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--reference_from_meta",
        action="store_true",
        help="use the first k-shot train entries in the dataset meta.json",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="root containing category/train|test/.../grounding_mask.png",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="write the unified results to this exact CSV path",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device
    k_shot = args.k_shot

    image_size = args.image_size
    save_path = os.path.join(args.save_path, dataset_name)
    if args.output_csv:
        output_csv = os.path.abspath(args.output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        txt_path = os.path.splitext(output_csv)[0] + ".log"
    else:
        os.makedirs(save_path, exist_ok=True)
        output_csv = os.path.join(save_path, "unified_results.csv")
        txt_path = os.path.join(save_path, "log.txt")

    reference_meta = None
    if args.reference_from_meta:
        with open(os.path.join(dataset_dir, "meta.json"), "r") as file:
            reference_meta = json.load(file)["train"]

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    UniVAD_model = UniVAD(image_size=args.image_size).to(device)

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    if dataset_name == "mvtec":
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "visa":
        test_data = VisaDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            mode="test",
        )
    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(
            root="./data/BrainMRI",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "his":
        test_data = HISDataset(
            root="./data/HIS",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "resc":
        test_data = RESCDataset(
            root="./data/RESC",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(
            root="./data/ChestXray",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(
            root="./data/OCT17",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(
            root="./data/LiverCT",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    else:
        raise NotImplementedError("Dataset not supported")

    test_dataloader = DataLoaderX(
        test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    results = {}
    results["cls_names"] = []
    results["imgs_masks"] = []
    results["anomaly_maps"] = []
    results["gt_sp"] = []
    results["pr_sp"] = []

    cls_last = None
    category_benchmarks = {}
    category_benchmark = None

    image_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    for items in tqdm(test_dataloader):
        image = items["img"].to(device)
        image_pil = items["img_pil"]
        image_path = items["img_path"][0]
        query_mask_path = None
        if args.mask_path:
            query_mask_path = get_grounding_mask_path(
                image_path, dataset_dir, args.mask_path
            )

        if args.class_name != "None":
            if args.class_name not in image_path:
                continue

        cls_name = items["cls_name"][0]
        results["cls_names"].append(cls_name)
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)  # px
        results["gt_sp"].append(items["anomaly"].item())

        if cls_name != cls_last:
            if category_benchmark is not None:
                category_benchmarks[cls_last] = category_benchmark.result()
            if reference_meta is not None:
                meta_cls_name = cls_name.replace(" ", "_")
                reference_entries = reference_meta[meta_cls_name]
                if len(reference_entries) < k_shot:
                    raise ValueError(
                        f"{meta_cls_name} has {len(reference_entries)} train references, "
                        f"but --k_shot={k_shot}"
                    )
                normal_image_paths = [
                    os.path.join(dataset_dir, entry["img_path"])
                    for entry in reference_entries[:k_shot]
                ]
            elif dataset_name == "mvtec":
                normal_image_paths = [
                    "./data/mvtec/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "mvtec_loco":
                normal_image_paths = [
                    "./data/mvtec_loco_caption/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "visa":
                if cls_name.replace(" ", "_") in [
                    "capsules",
                    "cashew",
                    "chewinggum",
                    "fryum",
                    "pipe_fryum",
                ]:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(3)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
                else:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(4)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
            elif dataset_name in [
                "his",
                "oct17",
                "chestxray",
                "brainmri",
                "liverct",
                "resc",
            ]:
                dir = (
                    "./data/"
                    + cls_name.replace(" ", "_")
                    + "/train/good"
                )
                files = sorted(os.listdir(dir))[:k_shot]
                normal_image_paths = [os.path.join(dir, file) for file in files]

            # normal_image_path = normal_image_paths[:k_shot]
            normal_images = torch.cat(
                [
                    image_transform(Image.open(x).convert("RGB")).unsqueeze(0)
                    for x in normal_image_paths
                ],
                dim=0,
            ).to(device)

            setup_data = {
                "few_shot_samples": normal_images,
                "dataset_category": cls_name.replace(" ", "_"),
                "image_path": normal_image_paths,
            }
            if args.mask_path:
                setup_data["mask_path"] = [
                    get_grounding_mask_path(path, dataset_dir, args.mask_path)
                    for path in normal_image_paths
                ]
            UniVAD_model.setup(setup_data)
            cls_last = cls_name
            category_benchmark = InferenceBenchmark(device)

        with torch.no_grad():
            category_benchmark.start_batch()
            pred_value = UniVAD_model(
                image, image_path, image_pil, grounding_mask_path=query_mask_path
            )
            anomaly_score, anomaly_map = (
                pred_value["pred_score"],
                pred_value["pred_mask"],
            )
            results["anomaly_maps"].append(anomaly_map.detach().cpu().numpy())
            overall_anomaly_score = anomaly_score.item()
            results["pr_sp"].append(overall_anomaly_score)
            category_benchmark.stop_batch(image.shape[0])

    if category_benchmark is not None:
        category_benchmarks[cls_last] = category_benchmark.result()

    evaluated_objects = list(dict.fromkeys(results["cls_names"]))

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    unified_rows = []

    threads = [None] * 20
    idx = 0
    for obj in tqdm(evaluated_objects):
        threads[idx] = threading.Thread(target=cal_score, args=(obj,))
        threads[idx].start()
        idx += 1

    for i in range(idx):
        threads[i].join()

    for obj in evaluated_objects:
        indices = [i for i, name in enumerate(results["cls_names"]) if name == obj]
        gt_px = np.asarray([results["imgs_masks"][i].squeeze(1).numpy() for i in indices])
        pr_px = np.asarray([results["anomaly_maps"][i] for i in indices])
        gt_sp = np.asarray([results["gt_sp"][i] for i in indices])
        pr_sp = np.asarray([results["pr_sp"][i] for i in indices])
        metrics = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)
        row = make_result_row(obj, metrics, category_benchmarks[obj])
        unified_rows.append(row)
        logger.info(format_result(row))

    # logger
    table_ls.append(
        [
            "mean",
            str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
            str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
        ]
    )
    results = tabulate(
        table_ls,
        headers=[
            "objects",
            "auroc_sp",
            "auroc_px",
        ],
        tablefmt="pipe",
    )
    logger.info("\n%s", results)
    unified_path = write_results_csv(output_csv, unified_rows)
    logger.info("Unified benchmark saved to %s", unified_path)
