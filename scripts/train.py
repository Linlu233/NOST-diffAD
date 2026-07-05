#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from nostdiffad.config import apply_overrides, load_yaml
from nostdiffad.data import IndustrialAnomalyDataset, SyntheticNormalDataset, build_mvtec_records, split_normal_train_val, split_test_records
from nostdiffad.engine import (
    build_feature_memory,
    evaluate_with_conformal,
    load_checkpoint,
    make_components,
    make_loader,
    save_checkpoint,
    train_one_epoch,
)
from nostdiffad.model import NOSTDiffAD
from nostdiffad.utils import ensure_dir, resolve_device, save_json, seed_everything


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def metric_value(metrics: dict | None, monitor: str) -> float | None:
    if not metrics or monitor not in metrics:
        return None
    value = float(metrics[monitor])
    if math.isnan(value):
        return None
    return value


def is_better(score: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return score > best + min_delta
    if mode == "min":
        return score < best - min_delta
    raise ValueError(f"Unknown early stopping mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke tests.")
    return parser.parse_args()


def build_datasets(config: dict, synthetic: bool):
    data_cfg = config["data"]
    if synthetic:
        train = SyntheticNormalDataset(length=8, image_size=int(data_cfg["image_size"]), anomalous=False)
        val = SyntheticNormalDataset(length=4, image_size=int(data_cfg["image_size"]), anomalous=False)
        test = SyntheticNormalDataset(length=6, image_size=int(data_cfg["image_size"]), anomalous=True)
        return train, val, test

    train_records_all = build_mvtec_records(
        data_cfg["root"],
        split="train",
        category=data_cfg.get("category"),
        mask_suffix=str(data_cfg.get("mask_suffix", ".png")),
        part_mask_root=data_cfg.get("part_mask_root"),
        part_mask_suffix=str(data_cfg.get("part_mask_suffix", ".png")),
    )
    train_records, val_records = split_normal_train_val(
        train_records_all,
        val_split=float(data_cfg["val_split"]),
        seed=int(config["seed"]),
        few_shot=data_cfg.get("few_shot", "full"),
    )
    test_records = build_mvtec_records(
        data_cfg["root"],
        split="test",
        category=data_cfg.get("category"),
        mask_suffix=str(data_cfg.get("mask_suffix", ".png")),
        part_mask_root=data_cfg.get("part_mask_root"),
        part_mask_suffix=str(data_cfg.get("part_mask_suffix", ".png")),
    )
    test_records = split_test_records(
        test_records,
        split_fraction=float(data_cfg.get("test_split_fraction", 1.0)),
        split_role=str(data_cfg.get("test_split_role", "all")),
        seed=int(data_cfg.get("test_split_seed", config["seed"])),
    )
    if not train_records:
        raise RuntimeError("No normal training records found. Check data.root and data.category.")
    if not val_records:
        raise RuntimeError("No normal validation records found for conformal calibration.")
    if not test_records:
        raise RuntimeError("No test records found. Check dataset layout.")
    image_size = int(data_cfg["image_size"])
    mask_resize_mode = str(data_cfg.get("mask_resize_mode", "nearest"))
    return (
        IndustrialAnomalyDataset(train_records, image_size, mask_resize_mode=mask_resize_mode),
        IndustrialAnomalyDataset(val_records, image_size, mask_resize_mode=mask_resize_mode),
        IndustrialAnomalyDataset(
            test_records,
            image_size,
            robustness=str(data_cfg.get("robustness", "none")),
            mask_resize_mode=mask_resize_mode,
        ),
    )


def main() -> None:
    args = parse_args()
    log(f"loading config {args.config}")
    config = apply_overrides(load_yaml(args.config), args.set)
    if args.synthetic:
        config["model"]["feature_extractor"] = "conv"
        config["model"]["feature_dim"] = int(config["model"].get("feature_dim", 384))
        config["train"]["epochs"] = min(int(config["train"]["epochs"]), 2)
        config["train"]["batch_size"] = min(int(config["train"]["batch_size"]), 2)
        config["model"]["num_classes"] = 1
        config["graph"]["use_mask_topology"] = False
        config["graph"]["beta_m"] = 0.0

    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))
    log(f"using device {device}")
    log("building datasets")
    train_set, val_set, test_set = build_datasets(config, args.synthetic)
    log(f"dataset sizes train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    train_loader = make_loader(train_set, int(config["train"]["batch_size"]), True, int(config["num_workers"]))
    val_loader = make_loader(val_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    test_loader = make_loader(test_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))

    grid = int(config["data"]["image_size"]) // int(config["model"]["patch_size"])
    max_nodes = grid * grid
    log("building model")
    model = NOSTDiffAD(config).to(device)
    log("building diffusion, prototype, nmf, energy components")
    schedule, prototype_bank, nmf, energy = make_components(config, device, max_nodes=max_nodes)
    feature_memory = build_feature_memory(model, train_loader, device, config)
    params = list(model.parameters()) + list(prototype_bank.parameters()) + list(nmf.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(config["train"]["lr"]), weight_decay=float(config["train"]["weight_decay"]))

    early_cfg = config["train"].get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    monitor = str(early_cfg.get("monitor", "image_auroc"))
    mode = str(early_cfg.get("mode", "max"))
    patience = int(early_cfg.get("patience", 0))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    min_epochs = int(early_cfg.get("min_epochs", 0))

    best: float | None = None
    best_epoch = 0
    best_metrics: dict | None = None
    history: list[dict] = []
    save_dir = ensure_dir(config["train"]["save_dir"])
    category = config["data"].get("category") or ("synthetic" if args.synthetic else "all")
    best_path = Path(save_dir) / f"{category}_best.pt"
    last_path = Path(save_dir) / f"{category}_last.pt"
    result_path = Path(config["eval"]["result_dir"]) / f"{category}_train_metrics.json"

    start_epoch = 1
    if bool(config["train"].get("resume", False)) and best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        best_metrics = best_state.get("metrics") or None
        best_score = metric_value(best_metrics, monitor)
        if best_score is not None:
            best = best_score
            best_epoch = int(best_state.get("epoch", 0))
            log(f"loaded best checkpoint epoch={best_epoch} {monitor}={best:.6f}")

    if bool(config["train"].get("resume", False)) and last_path.exists():
        state = load_checkpoint(last_path, model, prototype_bank, nmf, device)
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        log(f"resumed from {last_path} at epoch {start_epoch - 1}")

    if start_epoch > int(config["train"]["epochs"]):
        log(f"skip: start_epoch={start_epoch} exceeds train.epochs={config['train']['epochs']}")

    epochs_without_improvement = max(0, (start_epoch - 1) - best_epoch) if best_epoch else 0
    stopped_early = False
    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        log(f"epoch {epoch} train start")
        train_metrics = train_one_epoch(model, prototype_bank, nmf, schedule, train_loader, optimizer, config, device, epoch)
        log(f"epoch {epoch} eval start")
        eval_metrics, _ = evaluate_with_conformal(model, energy, val_loader, test_loader, config, device, feature_memory=feature_memory)
        score = metric_value(eval_metrics, monitor)
        save_checkpoint(last_path, model, prototype_bank, nmf, optimizer, config, epoch, eval_metrics)
        if score is not None and is_better(score, best, mode, min_delta):
            best = score
            best_epoch = epoch
            best_metrics = eval_metrics
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, prototype_bank, nmf, optimizer, config, epoch, eval_metrics)
            log(f"new best epoch={epoch} {monitor}={score:.6f}")
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        print({"epoch": epoch, "train": train_metrics, "eval": eval_metrics}, flush=True)
        if early_enabled and epoch >= min_epochs and epochs_without_improvement >= patience:
            log(
                "early stopping "
                f"epoch={epoch} best_epoch={best_epoch} {monitor}={best if best is not None else 'nan'} "
                f"patience={patience}"
            )
            stopped_early = True
            break

    ensure_dir(result_path.parent)
    save_json(
        {
            "completed": True,
            "completed_epoch": history[-1]["epoch"] if history else start_epoch - 1,
            "stopped_early": stopped_early,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_epoch": best_epoch,
            "best_eval": best_metrics,
            "latest_eval": history[-1]["eval"] if history else None,
            "resumed_from_epoch": start_epoch - 1,
            "early_stopping": {
                "enabled": early_enabled,
                "monitor": monitor,
                "mode": mode,
                "patience": patience,
                "min_delta": min_delta,
                "min_epochs": min_epochs,
            },
            "history": history,
        },
        result_path,
    )


if __name__ == "__main__":
    main()
