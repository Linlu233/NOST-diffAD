#!/usr/bin/env python
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from nostdiffad.data import IMAGE_EXTENSIONS, list_categories


SAM_CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/mvtec_ad")
    parser.add_argument("--output-root", default="datasets/part_masks/sam_vit_b_mvtec_ad")
    parser.add_argument("--category", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--model-type", choices=sorted(SAM_CHECKPOINTS), default="vit_b")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def download_checkpoint(model_type: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    filename, url = SAM_CHECKPOINTS[model_type]
    cache_dir = Path.home() / ".cache" / "nostdiffad" / "sam"
    path = cache_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"Downloading SAM checkpoint: {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bar.update(len(chunk))
    tmp_path.replace(path)
    return path


def iter_images(data_root: Path, category: str | None, splits: list[str]) -> list[Path]:
    images: list[Path] = []
    for cat in list_categories(data_root, category):
        for split in splits:
            split_dir = data_root / cat / split
            if not split_dir.exists():
                continue
            images.extend(sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS))
    return images


def output_path_for(image_path: Path, data_root: Path, output_root: Path) -> Path:
    relative = image_path.relative_to(data_root)
    return output_root / relative.with_suffix(".png")


def masks_to_label_map(masks: list[dict], size_hw: tuple[int, int]) -> np.ndarray:
    label_map = np.zeros(size_hw, dtype=np.uint16)
    sorted_masks = sorted(masks, key=lambda item: int(item.get("area", 0)), reverse=True)
    for idx, mask in enumerate(sorted_masks, start=1):
        if idx >= np.iinfo(np.uint16).max:
            break
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        label_map[segmentation] = idx
    return label_map


def save_label_map(label_map: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_map).save(output_path)


def valid_existing_mask(output_path: Path) -> bool:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False
    try:
        with Image.open(output_path) as image:
            image.verify()
    except Exception:
        return False
    return True


def main() -> None:
    args = parse_args()
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'segment_anything'. Install with: "
            "conda run -n nostdiffad python -m pip install segment-anything opencv-python-headless"
        ) from exc

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    device = resolve_device(args.device)
    checkpoint = download_checkpoint(args.model_type, args.checkpoint)
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    image_paths = iter_images(data_root, args.category, args.splits)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {data_root}")

    for image_path in tqdm(image_paths, desc="SAM part masks"):
        output_path = output_path_for(image_path, data_root, output_root)
        if not args.overwrite and valid_existing_mask(output_path):
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        masks = generator.generate(image)
        label_map = masks_to_label_map(masks, image.shape[:2])
        save_label_map(label_map, output_path)

    print(f"Wrote SAM part masks to {output_root}")


if __name__ == "__main__":
    main()
