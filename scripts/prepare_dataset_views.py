#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-realiad", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_file(source: Path, target: Path, overwrite: bool = False) -> bool:
    if not source.is_file():
        return False
    ensure_dir(target.parent)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source.resolve():
            return False
        if not overwrite:
            return False
        target.unlink()
    target.symlink_to(source.resolve())
    return True


def link_images(source_dir: Path, target_dir: Path, prefix: str = "", overwrite: bool = False) -> int:
    count = 0
    if not source_dir.exists():
        return count
    for source in sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        target_name = f"{prefix}{source.name}" if prefix else source.name
        count += int(link_file(source, target_dir / target_name, overwrite))
    return count


def prepare_mvtec_ad_2(root: Path, overwrite: bool) -> dict[str, int]:
    source_root = root / "mvtec_ad_2"
    target_root = root / "mvtec_ad_2_mvtec"
    stats = {"categories": 0, "links": 0}
    if not source_root.exists():
        return stats
    for category_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        category = category_dir.name
        stats["categories"] += 1
        stats["links"] += link_images(category_dir / "train" / "good", target_root / category / "train" / "good", overwrite=overwrite)
        stats["links"] += link_images(
            category_dir / "validation" / "good",
            target_root / category / "train" / "good",
            prefix="validation__",
            overwrite=overwrite,
        )
        stats["links"] += link_images(category_dir / "test_public" / "good", target_root / category / "test" / "good", overwrite=overwrite)
        stats["links"] += link_images(category_dir / "test_public" / "bad", target_root / category / "test" / "bad", overwrite=overwrite)
        stats["links"] += link_images(
            category_dir / "test_public" / "ground_truth" / "bad",
            target_root / category / "ground_truth" / "bad",
            overwrite=overwrite,
        )
    return stats


def prepare_mvtec_3d_rgb(root: Path, overwrite: bool) -> dict[str, int]:
    source_root = root / "mvtec_3d_anomaly_detection"
    target_root = root / "mvtec_3d_rgb_mvtec"
    stats = {"categories": 0, "links": 0}
    if not source_root.exists():
        return stats
    for category_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        category = category_dir.name
        stats["categories"] += 1
        stats["links"] += link_images(category_dir / "train" / "good" / "rgb", target_root / category / "train" / "good", overwrite=overwrite)
        stats["links"] += link_images(
            category_dir / "validation" / "good" / "rgb",
            target_root / category / "train" / "good",
            prefix="validation__",
            overwrite=overwrite,
        )
        test_root = category_dir / "test"
        if not test_root.exists():
            continue
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            defect = defect_dir.name
            stats["links"] += link_images(defect_dir / "rgb", target_root / category / "test" / defect, overwrite=overwrite)
            if defect != "good":
                stats["links"] += link_images(defect_dir / "gt", target_root / category / "ground_truth" / defect, overwrite=overwrite)
    return stats


def realiad_target_name(record: dict[str, str | None]) -> str:
    image_path = Path(str(record["image_path"]))
    return image_path.name


def prepare_realiad(root: Path, overwrite: bool) -> dict[str, int]:
    source_root = root / "realiad_1024"
    json_root = source_root / "realiad_jsons"
    target_root = root / "realiad_1024_mvtec"
    stats = {"categories": 0, "links": 0, "missing": 0}
    if not json_root.exists():
        return stats
    for json_path in sorted(json_root.glob("*.json")):
        category = json_path.stem
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        prefix = str(payload.get("meta", {}).get("prefix", f"{category}/"))
        category_source = source_root / prefix
        stats["categories"] += 1
        for split_name, target_split in [("train", "train"), ("test", "test")]:
            for record in payload.get(split_name, []):
                anomaly_class = str(record["anomaly_class"])
                defect = "good" if anomaly_class == "OK" else anomaly_class
                source_image = category_source / str(record["image_path"])
                target_image = target_root / category / target_split / defect / realiad_target_name(record)
                if link_file(source_image, target_image, overwrite):
                    stats["links"] += 1
                elif not source_image.is_file():
                    stats["missing"] += 1
                mask_path = record.get("mask_path")
                if target_split == "test" and defect != "good" and mask_path:
                    source_mask = category_source / str(mask_path)
                    target_mask = target_root / category / "ground_truth" / defect / Path(str(mask_path)).name
                    if link_file(source_mask, target_mask, overwrite):
                        stats["links"] += 1
                    elif not source_mask.is_file():
                        stats["missing"] += 1
    return stats


def prepare_visa(root: Path, overwrite: bool) -> dict[str, int]:
    source_root = root / "visa"
    csv_path = source_root / "split_csv" / "1cls.csv"
    target_root = root / "visa_mvtec"
    stats = {"categories": 0, "links": 0, "missing": 0}
    if not csv_path.exists():
        return stats

    categories: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            category = str(record["object"])
            categories.add(category)
            split = str(record["split"])
            label = str(record["label"])
            defect = "good" if label == "normal" else "anomaly"
            source_image = source_root / str(record["image"])
            target_image = target_root / category / split / defect / source_image.name
            if link_file(source_image, target_image, overwrite):
                stats["links"] += 1
            elif not source_image.is_file():
                stats["missing"] += 1

            mask = str(record.get("mask") or "")
            if split == "test" and defect != "good" and mask:
                source_mask = source_root / mask
                target_mask = target_root / category / "ground_truth" / defect / source_mask.name
                if link_file(source_mask, target_mask, overwrite):
                    stats["links"] += 1
                elif not source_mask.is_file():
                    stats["missing"] += 1
    stats["categories"] = len(categories)
    return stats


def main() -> None:
    args = parse_args()
    root = Path(args.datasets_root)
    results = {
        "mvtec_ad_2_mvtec": prepare_mvtec_ad_2(root, args.overwrite),
        "mvtec_3d_rgb_mvtec": prepare_mvtec_3d_rgb(root, args.overwrite),
        "visa_mvtec": prepare_visa(root, args.overwrite),
    }
    if not args.skip_realiad:
        results["realiad_1024_mvtec"] = prepare_realiad(root, args.overwrite)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
