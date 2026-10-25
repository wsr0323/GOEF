import argparse
import json
from pathlib import Path

from models.component_segmentaion import grounding_segmentation
import yaml


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def read_yaml(path):
    with path.open("r") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def expected_mask_path(image_path, dataset_root, mask_root):
    relative_path = image_path.relative_to(dataset_root).with_suffix("")
    return mask_root / relative_path / "grounding_mask.png"


def collect_images(category_root):
    images = []
    for split in ("train", "test"):
        split_root = category_root / split
        if not split_root.exists():
            continue
        images.extend(
            path
            for path in split_root.glob("*/*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    return sorted(images)


def prepare_dataset(dataset_root, mask_root, config_root, skip_existing):
    with (dataset_root / "meta.json").open("r") as file:
        categories = list(json.load(file)["test"].keys())

    for category in categories:
        images = collect_images(dataset_root / category)
        if skip_existing:
            images = [
                path
                for path in images
                if not expected_mask_path(
                    path, dataset_root, mask_root
                ).is_file()
            ]
        if not images:
            print(f"[{dataset_root.name}] {category}: masks already complete")
            continue

        config = read_yaml(config_root / f"{category}.yaml")["grounding_config"]
        output_dir = mask_root / category
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{dataset_root.name}] {category}: processing {len(images)} images")
        grounding_segmentation(
            [str(path) for path in images], str(output_dir), config
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate reusable component masks for few-shot MVTec and VisA data."
    )
    parser.add_argument(
        "--source_root",
        default=os.environ.get("GOEF_CENTER_ROOT", "data") + "/5-shot/centers_01_02_03_04_05",
        help="5-shot root containing the mvtec and visa directories",
    )
    parser.add_argument("--mask_root", default="./masks/fewshot")
    parser.add_argument("--config_root", default="./configs/class_histogram")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    config_root = Path(args.config_root).resolve()
    for dataset in ("mvtec", "visa"):
        prepare_dataset(
            source_root / dataset,
            mask_root / dataset,
            config_root,
            args.skip_existing,
        )


if __name__ == "__main__":
    main()
