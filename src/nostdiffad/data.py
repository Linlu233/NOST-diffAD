from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMAL_DEFECT_TYPES = {"good", "ok"}


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: int
    category: str
    defect_type: str
    anomaly_mask_path: Path | None = None
    part_mask_path: Path | None = None


def list_categories(root: str | Path, category: str | None = None) -> list[str]:
    root_path = Path(root)
    if category is not None:
        return [str(category)]
    return sorted(path.name for path in root_path.iterdir() if path.is_dir())


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _mask_path_for(base_dir: Path, image_path: Path, mask_suffix: str) -> Path | None:
    candidates = [
        base_dir / f"{image_path.stem}_mask{image_path.suffix}",
        base_dir / f"{image_path.stem}{mask_suffix}",
        base_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in base_dir.glob(f"{image_path.stem}*") if path.is_file())
    if matches:
        return matches[0]
    for directory in sorted(path for path in base_dir.glob(f"{image_path.stem}*") if path.is_dir()):
        nested = _mask_path_for(directory, image_path, mask_suffix)
        if nested is not None:
            return nested
        nested_images = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if nested_images:
            return nested_images[0]
    return None


def _anomaly_mask_path_for(root: Path, category: str, defect_type: str, image_path: Path, mask_suffix: str) -> Path | None:
    if defect_type in NORMAL_DEFECT_TYPES:
        return None
    gt_dir = root / category / "ground_truth" / defect_type
    if not gt_dir.exists():
        return None
    return _mask_path_for(gt_dir, image_path, mask_suffix)


def _part_mask_path_for(
    part_mask_root: Path | None,
    category: str,
    split: str,
    defect_type: str,
    image_path: Path,
    part_mask_suffix: str,
) -> Path | None:
    if part_mask_root is None:
        return None
    candidates = [
        part_mask_root / category / split / defect_type,
        part_mask_root / category / defect_type,
        part_mask_root / category,
        part_mask_root,
    ]
    for base_dir in candidates:
        if base_dir.exists():
            found = _mask_path_for(base_dir, image_path, part_mask_suffix)
            if found is not None:
                return found
    return None


def build_mvtec_records(
    root: str | Path,
    split: str,
    category: str | None = None,
    mask_suffix: str = ".png",
    part_mask_root: str | Path | None = None,
    part_mask_suffix: str = ".png",
) -> list[SampleRecord]:
    root_path = Path(root)
    part_root_path = Path(part_mask_root) if part_mask_root else None
    records: list[SampleRecord] = []
    for cat in list_categories(root_path, category):
        split_dir = root_path / cat / split
        if not split_dir.exists():
            continue
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type in NORMAL_DEFECT_TYPES else 1
            for image_path in _list_images(defect_dir):
                anomaly_mask_path = _anomaly_mask_path_for(root_path, cat, defect_type, image_path, mask_suffix)
                part_mask_path = _part_mask_path_for(part_root_path, cat, split, defect_type, image_path, part_mask_suffix)
                records.append(SampleRecord(image_path, label, cat, defect_type, anomaly_mask_path, part_mask_path))
    return records


def split_normal_train_val(
    records: list[SampleRecord],
    val_split: float,
    seed: int,
    few_shot: int | str = "full",
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    normal_records = [record for record in records if record.label == 0]
    by_category: dict[str, list[SampleRecord]] = {}
    for record in normal_records:
        by_category.setdefault(record.category, []).append(record)

    train: list[SampleRecord] = []
    val: list[SampleRecord] = []
    rng = random.Random(seed)
    for category_records in by_category.values():
        shuffled = list(category_records)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
        val_part = shuffled[:val_count]
        train_part = shuffled[val_count:]
        if few_shot != "full":
            shot = int(few_shot)
            train_part = train_part[:shot]
        train.extend(train_part)
        val.extend(val_part)
    return train, val


def split_test_records(
    records: list[SampleRecord],
    split_fraction: float,
    split_role: str,
    seed: int,
) -> list[SampleRecord]:
    """Deterministically split test records by label and defect type."""

    if split_fraction >= 1.0 or split_role == "all":
        return records
    if split_role not in {"tune", "final"}:
        raise ValueError(f"Unknown test split role: {split_role}")

    rng = random.Random(seed)
    by_label_type: dict[tuple[int, str], list[SampleRecord]] = {}
    for record in records:
        by_label_type.setdefault((record.label, record.defect_type), []).append(record)

    selected: list[SampleRecord] = []
    held_out: list[SampleRecord] = []
    for grouped_records in by_label_type.values():
        shuffled = list(grouped_records)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * split_fraction))) if len(shuffled) > 1 else len(shuffled)
        selected.extend(shuffled[:count])
        held_out.extend(shuffled[count:])
    return selected if split_role == "tune" else held_out


def image_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def mask_transform(image_size: int, resize_mode: str = "nearest") -> Callable[[Image.Image], torch.Tensor]:
    if resize_mode == "nearest":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.PILToTensor(),
            ]
        )
    if resize_mode == "max":
        def transform(mask_img: Image.Image) -> torch.Tensor:
            resample = Image.Resampling.BOX if min(mask_img.size) > image_size else Image.Resampling.NEAREST
            mask_array = np.array(mask_img.resize((image_size, image_size), resample=resample), copy=True)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            return torch.from_numpy(mask_array).unsqueeze(0)

        return transform
    raise ValueError(f"Unknown mask resize mode: {resize_mode}")


def load_label_mask(path: str | Path, image_size: int) -> torch.Tensor:
    mask_img = Image.open(path)
    mask_img = mask_img.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    mask_array = np.asarray(mask_img)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    mask_array = np.asarray(mask_array, dtype=np.int64)
    return torch.from_numpy(mask_array).unsqueeze(0).long()


class IndustrialAnomalyDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: int,
        robustness: str = "none",
        mask_resize_mode: str = "nearest",
    ) -> None:
        self.records = records
        self.image_transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size, mask_resize_mode)
        self.image_size = image_size
        self.robustness = robustness
        categories = sorted({record.category for record in records})
        self.category_to_id = {category: idx for idx, category in enumerate(categories)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        image_tensor = apply_robustness(image_tensor, self.robustness)

        if record.anomaly_mask_path and record.anomaly_mask_path.exists():
            mask_img = Image.open(record.anomaly_mask_path).convert("L")
            mask = self.mask_transform(mask_img).float()
            mask = (mask > 0).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        if record.part_mask_path and record.part_mask_path.exists():
            part_mask = load_label_mask(record.part_mask_path, self.image_size)
            part_mask_available = torch.tensor(True)
        else:
            part_mask = torch.full((1, self.image_size, self.image_size), -1, dtype=torch.long)
            part_mask_available = torch.tensor(False)

        return {
            "image": image_tensor,
            "mask": mask,
            "part_mask": part_mask,
            "part_mask_available": part_mask_available,
            "label": torch.tensor(record.label, dtype=torch.long),
            "category_id": torch.tensor(self.category_to_id[record.category], dtype=torch.long),
            "category": record.category,
            "defect_type": record.defect_type,
            "path": str(record.image_path),
        }


class SyntheticNormalDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    """Small deterministic dataset for smoke tests when benchmarks are absent."""

    def __init__(self, length: int = 8, image_size: int = 224, anomalous: bool = False) -> None:
        self.length = length
        self.image_size = image_size
        self.anomalous = anomalous

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        rng = np.random.default_rng(index)
        image = rng.normal(0.0, 0.08, size=(3, self.image_size, self.image_size)).astype("float32")
        yy, xx = np.mgrid[0 : self.image_size, 0 : self.image_size]
        image += np.sin(xx / 11.0)[None] * 0.2 + np.cos(yy / 17.0)[None] * 0.2
        mask = np.zeros((1, self.image_size, self.image_size), dtype="float32")
        label = 0
        if self.anomalous and index % 2 == 1:
            y0, x0 = self.image_size // 3, self.image_size // 2
            image[:, y0 : y0 + 24, x0 : x0 + 24] += 2.0
            mask[:, y0 : y0 + 24, x0 : x0 + 24] = 1.0
            label = 1
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "part_mask": torch.full((1, self.image_size, self.image_size), -1, dtype=torch.long),
            "part_mask_available": torch.tensor(False),
            "label": torch.tensor(label, dtype=torch.long),
            "category_id": torch.tensor(0, dtype=torch.long),
            "category": "synthetic",
            "defect_type": "synthetic_anomaly" if label else "good",
            "path": f"synthetic_{index}.png",
        }


def collate_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str]]:
    tensor_keys = ["image", "mask", "part_mask", "part_mask_available", "label", "category_id"]
    out: dict[str, torch.Tensor | list[str]] = {}
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch])  # type: ignore[arg-type]
    for key in ["category", "defect_type", "path"]:
        out[key] = [str(item[key]) for item in batch]
    return out


def apply_robustness(image: torch.Tensor, robustness: str) -> torch.Tensor:
    if robustness == "none":
        return image
    if robustness == "brightness":
        return image * 1.25
    if robustness == "gaussian_noise":
        generator = torch.Generator(device=image.device)
        generator.manual_seed(0)
        return image + torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * 0.05
    if robustness == "jpeg_compression":
        quantized = torch.round(image * 16.0) / 16.0
        return quantized
    raise ValueError(f"Unknown robustness setting: {robustness}")
