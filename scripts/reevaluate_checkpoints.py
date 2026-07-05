#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nostdiffad.config import apply_overrides
from nostdiffad.engine import evaluate_with_conformal, load_checkpoint, make_components, make_loader
from nostdiffad.model import NOSTDiffAD
from nostdiffad.utils import ensure_dir, resolve_device, seed_everything
from train import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/diagnostics/reevaluated_checkpoints.json")
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def checkpoint_config(path: Path, device: torch.device, overrides: list[str]) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    config = state.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"Checkpoint has no config dict: {path}")
    config = apply_overrides(config, ["device=" + str(device), *overrides])
    return config


def reevaluate(path: Path, device: torch.device, overrides: list[str]) -> dict[str, Any]:
    config = checkpoint_config(path, device, overrides)
    seed_everything(int(config["seed"]))
    _, val_set, test_set = build_datasets(config, synthetic=False)
    val_loader = make_loader(val_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    test_loader = make_loader(test_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    grid = int(config["data"]["image_size"]) // int(config["model"]["patch_size"])

    model = NOSTDiffAD(config).to(device)
    schedule, prototype_bank, nmf, energy = make_components(config, device, max_nodes=grid * grid)
    load_checkpoint(path, model, prototype_bank, nmf, device)
    metrics, threshold = evaluate_with_conformal(model, energy, val_loader, test_loader, config, device)
    _ = schedule
    return {
        "checkpoint": str(path),
        "dataset_root": str(config["data"]["root"]),
        "category": str(config["data"].get("category")),
        "image_size": int(config["data"]["image_size"]),
        "upsample_mode": str(config["energy"].get("upsample_mode", "bilinear")),
        "energy": {
            "alpha": float(config["energy"]["alpha"]),
            "beta": float(config["energy"]["beta"]),
            "gamma": float(config["energy"]["gamma"]),
            "eta": float(config["energy"]["eta"]),
            "topk_ratio": float(config["energy"]["topk_ratio"]),
        },
        "threshold": threshold,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = [reevaluate(Path(path), device, list(args.set)) for path in args.checkpoints]
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
