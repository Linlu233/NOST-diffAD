#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from nostdiffad.config import apply_overrides, load_yaml
from nostdiffad.engine import build_feature_memory, evaluate_with_conformal, make_components, make_loader
from nostdiffad.model import NOSTDiffAD
from nostdiffad.utils import ensure_dir, resolve_device, save_json, seed_everything

from train import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args.set)
    if args.synthetic:
        config["model"]["feature_extractor"] = "conv"
        config["model"]["num_classes"] = 1
        config["graph"]["use_mask_topology"] = False
        config["graph"]["beta_m"] = 0.0
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))
    train_set, val_set, test_set = build_datasets(config, args.synthetic)
    train_loader = make_loader(train_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    val_loader = make_loader(val_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    test_loader = make_loader(test_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))

    grid = int(config["data"]["image_size"]) // int(config["model"]["patch_size"])
    max_nodes = grid * grid
    model = NOSTDiffAD(config).to(device)
    _, _, _, energy = make_components(config, device, max_nodes=max_nodes)
    feature_memory = build_feature_memory(model, train_loader, device, config)
    metrics, threshold = evaluate_with_conformal(
        model,
        energy,
        val_loader,
        test_loader,
        config,
        device,
        feature_memory=feature_memory,
    )
    metrics["conformal_threshold"] = threshold
    result_dir = ensure_dir(config["eval"]["result_dir"])
    category = config["data"].get("category") or ("synthetic" if args.synthetic else "all")
    result_path = Path(result_dir) / f"{category}_train_metrics.json"
    save_json(
        {
            "completed": True,
            "best_epoch": 0,
            "best_eval": metrics,
            "latest_eval": metrics,
            "method": "feature_memory",
        },
        result_path,
    )
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
