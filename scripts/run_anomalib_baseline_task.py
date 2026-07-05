#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Cfa, Cflow, Dfkde, Dinomaly, Fastflow, Padim, Patchcore, ReverseDistillation, Stfpm, WinClip

from nostdiffad.data import SampleRecord, build_mvtec_records
from nostdiffad.metrics import compute_metrics
from nostdiffad.utils import ensure_dir, save_json, seed_everything


NORMAL_DIRS = {"good", "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--limit-test-batches", type=float, default=1.0)
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_empty_mask(image_path: Path, mask_path: Path) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path)
    Image.new("L", image.size, color=0).save(mask_path)


def add_records_to_view(records: list[SampleRecord], image_dir: Path, mask_dir: Path | None = None) -> None:
    for idx, record in enumerate(records):
        image_name = f"{idx:06d}_{record.defect_type}{record.image_path.suffix.lower()}"
        image_out = image_dir / image_name
        link_or_copy(record.image_path, image_out)
        if mask_dir is not None:
            mask_out = mask_dir / f"{Path(image_name).stem}.png"
            if record.anomaly_mask_path and record.anomaly_mask_path.exists():
                link_or_copy(record.anomaly_mask_path, mask_out)
            else:
                write_empty_mask(record.image_path, mask_out)


def prepare_anomalib_view(root: Path, category: str, view_root: Path) -> tuple[str, str | None, str, str]:
    """Create a flat Folder datamodule view with one abnormal image per mask.

    Anomalib's Folder datamodule assumes that all abnormal images and masks can be
    matched by flat filename ordering. Several converted datasets here contain nested
    mask folders or training split anomalies, so the direct paths are not robust enough.
    """

    if view_root.exists():
        shutil.rmtree(view_root)
    train_records = build_mvtec_records(root, "train", category)
    test_records = build_mvtec_records(root, "test", category)
    normal_train = [record for record in train_records if record.label == 0]
    normal_test = [record for record in test_records if record.label == 0]
    abnormal_test = [record for record in test_records if record.label == 1]
    if not normal_train:
        raise FileNotFoundError(f"No normal train samples found for {root}/{category}.")
    if not abnormal_test:
        raise FileNotFoundError(f"No abnormal test samples found for {root}/{category}.")

    train_good = view_root / "train" / "good"
    test_good = view_root / "test" / "good"
    test_defect = view_root / "test" / "defect"
    gt_defect = view_root / "ground_truth" / "defect"
    add_records_to_view(normal_train, train_good)
    if normal_test:
        add_records_to_view(normal_test, test_good)
    add_records_to_view(abnormal_test, test_defect, gt_defect)
    return "train/good", "test/good" if normal_test else None, "test/defect", "ground_truth/defect"


def make_model(method: str, category: str) -> Any:
    key = method.lower()
    common = {"visualizer": False}
    if key == "patchcore":
        return Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True, **common)
    if key == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True, **common)
    if key == "fastflow":
        return Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8, **common)
    if key == "reverse_distillation":
        return ReverseDistillation(backbone="wide_resnet50_2", pre_trained=True, **common)
    if key == "stfpm":
        return Stfpm(backbone="resnet18", **common)
    if key == "cfa":
        return Cfa(backbone="wide_resnet50_2", **common)
    if key == "cflow":
        return Cflow(backbone="wide_resnet50_2", pre_trained=True, **common)
    if key == "dfkde":
        return Dfkde(backbone="resnet18", pre_trained=True, **common)
    if key == "dinomaly":
        return Dinomaly(encoder_name="dinov2_vit_small_14", **common)
    if key == "winclip":
        return WinClip(class_name=category.replace("_", " "), **common)
    raise ValueError(f"Unknown baseline method: {method}")


def default_epochs(method: str) -> int:
    key = method.lower()
    if key in {"patchcore", "padim", "dfkde", "winclip"}:
        return 1
    if key in {"fastflow", "cflow", "stfpm", "reverse_distillation", "cfa"}:
        return 30
    if key == "dinomaly":
        return 50
    return 20


def flatten_predictions(predictions: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    image_score_map_proxy = False
    for batch in predictions:
        gt_label = torch.as_tensor(batch.gt_label).detach().cpu().reshape(-1).numpy().astype(np.int64)
        pred_score = torch.as_tensor(batch.pred_score).detach().cpu().reshape(-1).numpy().astype(np.float64)
        gt_mask = torch.as_tensor(batch.gt_mask).detach().cpu().numpy()
        if gt_mask.ndim == 4 and gt_mask.shape[1] == 1:
            gt_mask = gt_mask[:, 0]
        anomaly_map_value = getattr(batch, "anomaly_map", None)
        if anomaly_map_value is None:
            image_score_map_proxy = True
            anomaly_map = np.broadcast_to(pred_score.reshape(-1, 1, 1), gt_mask.shape).astype(np.float64)
        else:
            anomaly_map = torch.as_tensor(anomaly_map_value).detach().cpu().numpy().astype(np.float64)
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map[:, 0]
        labels.append(gt_label)
        scores.append(pred_score)
        masks.append((gt_mask > 0).astype(np.uint8))
        maps.append(anomaly_map)
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(masks), np.concatenate(maps), image_score_map_proxy


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        try:
            number = float(value)
            out[key] = number if math.isfinite(number) else None
        except Exception:
            out[key] = None
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    result_dir = ensure_dir(args.result_dir)
    result_path = result_dir / f"{args.category}_train_metrics.json"
    work_dir = ensure_dir(args.work_dir)
    view_root = work_dir / "input_view"
    normal_train, normal_test, abnormal, mask_dir = prepare_anomalib_view(Path(args.root), args.category, view_root)
    datamodule = Folder(
        name=f"{args.dataset}_{args.category}",
        root=view_root,
        normal_dir=normal_train,
        normal_test_dir=normal_test,
        abnormal_dir=abnormal,
        mask_dir=mask_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_split_mode="from_train",
        val_split_ratio=0.15,
        seed=args.seed,
    )
    model = make_model(args.method, args.category)
    max_epochs = int(args.max_epochs or default_epochs(args.method))
    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=False,
        max_epochs=max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        default_root_dir=work_dir,
    )
    test_metrics = engine.train(model=model, datamodule=datamodule)
    predict_start = time.perf_counter()
    predictions = engine.predict(model=model, datamodule=datamodule, return_predictions=True)
    if predictions is None:
        raise RuntimeError("Anomalib did not return predictions.")
    image_labels, image_scores, masks, score_maps, image_score_map_proxy = flatten_predictions(predictions)
    elapsed = time.perf_counter() - predict_start
    unified = compute_metrics(
        image_labels=image_labels,
        image_scores=image_scores,
        masks=masks,
        score_maps=score_maps,
        threshold=None,
        inference_seconds=elapsed,
    )
    unified = finite_metrics(unified)
    payload = {
        "completed": True,
        "method": args.method,
        "dataset": args.dataset,
        "category": args.category,
        "root": args.root,
        "image_score_map_proxy": image_score_map_proxy,
        "max_epochs": max_epochs,
        "anomalib_test_metrics": test_metrics,
        "best_epoch": 0,
        "completed_epoch": max_epochs,
        "best_eval": unified,
        "latest_eval": unified,
        "history": [{"epoch": max_epochs, "eval": unified}],
    }
    save_json(payload, result_path)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
