#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from nostdiffad.config import load_yaml, set_by_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--cross-category", action="store_true")
    parser.add_argument("--output", default="outputs/experiment_commands.sh")
    return parser.parse_args()


def apply_ablation(base: dict, settings: dict) -> dict:
    cfg = deepcopy(base)
    for key, value in settings.items():
        if key == "description":
            continue
        if "." in key:
            set_by_path(cfg, key, value)
        else:
            cfg[key] = value
    return cfg


def main() -> None:
    args = parse_args()
    base = load_yaml(args.config)
    matrix = load_yaml(args.matrix)
    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    categories = [args.category] if args.category else ["<CATEGORY>"]
    for dataset in matrix["datasets"]:
        for category in categories:
            for ablation, settings in matrix["ablations"].items():
                for few_shot in matrix["few_shot"]:
                    for robustness in matrix["robustness"]:
                        cfg = apply_ablation(base, settings or {})
                        cfg["data"]["root"] = args.data_root
                        cfg["data"]["dataset"] = dataset
                        cfg["data"]["category"] = None if args.cross_category else category
                        cfg["data"]["few_shot"] = few_shot
                        cfg["data"]["robustness"] = robustness
                        category_name = "all_categories" if args.cross_category else category
                        cfg_path = Path("outputs/generated_configs") / (
                            f"{dataset.replace(' ', '_')}_{category_name}_{ablation}_shot{few_shot}_robust{robustness}.yaml"
                        )
                        cfg_path.parent.mkdir(parents=True, exist_ok=True)
                        with cfg_path.open("w", encoding="utf-8") as handle:
                            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
                        commands.append(f"python scripts/train.py --config {cfg_path}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
