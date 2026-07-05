#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="datasets/kolektorsdd2")
    parser.add_argument("--output-root", default="datasets/kolektorsdd2_mvtec")
    parser.add_argument("--category", default="kolektorsdd2")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def is_anomalous(mask_path: Path | None) -> bool:
    if mask_path is None or not mask_path.exists():
        return False
    mask = np.asarray(Image.open(mask_path))
    return bool(mask.max() > 0)


def link_or_copy(source: Path, target: Path, copy_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def convert_split(source_root: Path, output_root: Path, category: str, split: str, copy_mode: str) -> dict[str, int]:
    stats = {"images": 0, "good": 0, "defect": 0, "missing_gt": 0}
    split_dir = source_root / split
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    for image_path in sorted(split_dir.glob("*.png")):
        if image_path.name.endswith("_GT.png"):
            continue
        stats["images"] += 1
        mask_path = image_path.with_name(f"{image_path.stem}_GT.png")
        if not mask_path.exists():
            stats["missing_gt"] += 1
            mask_path = None

        defect_type = "defect" if is_anomalous(mask_path) else "good"
        stats[defect_type] += 1

        target_image = output_root / category / split / defect_type / image_path.name
        link_or_copy(image_path, target_image, copy_mode)

        if defect_type != "good" and mask_path is not None:
            target_mask = output_root / category / "ground_truth" / defect_type / image_path.name
            link_or_copy(mask_path, target_mask, copy_mode)

    return stats


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    stats = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "category": args.category,
        "splits": {
            split: convert_split(source_root, output_root, args.category, split, args.copy_mode)
            for split in ["train", "test"]
        },
    }
    manifest_path = output_root / args.category / "conversion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
