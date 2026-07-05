#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from nostdiffad.config import apply_overrides, load_yaml
from nostdiffad.data import IndustrialAnomalyDataset, SyntheticNormalDataset, build_mvtec_records, split_normal_train_val, split_test_records
from nostdiffad.engine import build_feature_memory, evaluate_with_conformal, load_checkpoint, make_components, make_loader
from nostdiffad.model import NOSTDiffAD
from nostdiffad.utils import ensure_dir, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def build_eval_sets(config: dict, synthetic: bool):
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
    config = apply_overrides(load_yaml(args.config), args.set)
    if args.synthetic:
        config["model"]["feature_extractor"] = "conv"
        config["model"]["num_classes"] = 1
        config["graph"]["use_mask_topology"] = False
        config["graph"]["beta_m"] = 0.0
    checkpoint = config["eval"].get("checkpoint")
    if not checkpoint:
        raise RuntimeError("Set eval.checkpoint=/path/to/checkpoint.pt")
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))
    train_set, val_set, test_set = build_eval_sets(config, args.synthetic)
    train_loader = make_loader(train_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    val_loader = make_loader(val_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    test_loader = make_loader(test_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    grid = int(config["data"]["image_size"]) // int(config["model"]["patch_size"])
    max_nodes = grid * grid
    model = NOSTDiffAD(config).to(device)
    _, prototype_bank, nmf, energy = make_components(config, device, max_nodes=max_nodes)
    load_checkpoint(checkpoint, model, prototype_bank, nmf, device)
    feature_memory = build_feature_memory(model, train_loader, device, config)
    metrics, threshold = evaluate_with_conformal(model, energy, val_loader, test_loader, config, device, feature_memory=feature_memory)
    metrics["conformal_threshold"] = threshold
    result_dir = ensure_dir(config["eval"]["result_dir"])
    category = config["data"].get("category") or ("synthetic" if args.synthetic else "all")
    result_path = Path(result_dir) / f"{category}_eval_metrics.json"
    save_json(metrics, result_path)
    print(metrics)


if __name__ == "__main__":
    main()
