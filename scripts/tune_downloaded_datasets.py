#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nostdiffad.config import load_yaml
from run_downloaded_official_experiments import Experiment, downloaded_experiments, mask_status, quoted_override, shell_join


@dataclass(frozen=True)
class Trial:
    name: str
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--output-root", default="outputs/hparam_tuning")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--tune-test-fraction", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="selection_score")
    parser.add_argument("--write-bash", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def trials() -> list[Trial]:
    return [
        Trial("base", {}),
        Trial("lr3e-5", {"train.lr": 3e-5}),
        Trial("lr3e-4", {"train.lr": 3e-4}),
        Trial("proto8_rank4", {"model.num_prototypes": 8, "model.nmf_rank": 4}),
        Trial("proto32_rank8", {"model.num_prototypes": 32, "model.nmf_rank": 8}),
        Trial("energy_proto_high", {"energy.alpha": 0.5, "energy.beta": 2.0, "energy.gamma": 0.5, "energy.eta": 0.5}),
        Trial("energy_score_high", {"energy.alpha": 2.0, "energy.beta": 1.0, "energy.gamma": 0.5, "energy.eta": 0.5}),
        Trial("score_proto_only", {"energy.gamma": 0.0, "energy.eta": 0.0}),
        Trial("no_topo", {"energy.gamma": 0.0}),
        Trial("no_wave", {"energy.eta": 0.0}),
        Trial("topo_tiny", {"energy.gamma": 0.001, "energy.eta": 0.0}),
        Trial("topo_small", {"energy.gamma": 0.01, "energy.eta": 0.0}),
        Trial("score_proto_topo_tiny", {"energy.alpha": 1.0, "energy.beta": 1.0, "energy.gamma": 0.001, "energy.eta": 0.0}),
        Trial("proto_only", {"energy.alpha": 0.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0}),
        Trial("proto_only_topk001", {"energy.alpha": 0.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0, "energy.topk_ratio": 0.01}),
        Trial("proto_only_topk010", {"energy.alpha": 0.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0, "energy.topk_ratio": 0.10}),
        Trial("score_only", {"energy.alpha": 1.0, "energy.beta": 0.0, "energy.gamma": 0.0, "energy.eta": 0.0}),
        Trial("proto_high_no_topo_wave", {"energy.alpha": 0.5, "energy.beta": 2.0, "energy.gamma": 0.0, "energy.eta": 0.0}),
        Trial("score_high_no_topo_wave", {"energy.alpha": 2.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0}),
        Trial("topk02", {"energy.topk_ratio": 0.02}),
        Trial("topk10", {"energy.topk_ratio": 0.10}),
        Trial("graph_sigma08", {"graph.sigma_p": 0.08}),
        Trial("graph_sigma25", {"graph.sigma_p": 0.25}),
        Trial("lap_low", {"loss.lambda_lap": 0.0001}),
        Trial("lap_high", {"loss.lambda_lap": 0.005}),
    ]


def dataset_groups(experiments: list[Experiment]) -> dict[str, list[Experiment]]:
    groups: dict[str, list[Experiment]] = {}
    for experiment in experiments:
        groups.setdefault(experiment.dataset, []).append(experiment)
    return groups


def representative_experiments(groups: dict[str, list[Experiment]]) -> list[Experiment]:
    out: list[Experiment] = []
    for dataset, experiments in sorted(groups.items()):
        by_name = {experiment.category: experiment for experiment in experiments}
        if dataset == "mvtec_ad":
            preferred = ["bottle", "cable", "hazelnut", "screw", "transistor"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        elif dataset == "btad":
            out.extend(experiments)
        elif dataset == "mpdd":
            preferred = ["bracket_black", "connector", "metal_plate", "tubes"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        elif dataset == "mvtec_loco":
            out.extend(experiments)
        elif dataset == "mvtec_ad_2":
            preferred = ["can", "fabric", "vial", "wallplugs"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        elif dataset == "mvtec_3d_rgb":
            preferred = ["bagel", "cable_gland", "cookie", "foam", "tire"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        elif dataset == "realiad_1024":
            preferred = ["audiojack", "bottle_cap", "pcb", "usb", "zipper"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        elif dataset == "visa":
            preferred = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "pcb1", "pipe_fryum"]
            out.extend(by_name[name] for name in preferred if name in by_name)
        else:
            out.extend(experiments)
    return out


def metric_from_result(path: Path, metric: str) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return float("nan")
    if not payload.get("completed"):
        return float("nan")
    best = payload.get("best_eval") or {}
    value = best.get(metric)
    return float(value) if value is not None else float("nan")


def write_best_config(base_config: dict[str, Any], dataset: str, best_trial: Trial, output_root: Path) -> Path:
    config = json.loads(json.dumps(base_config))
    for key, value in best_trial.overrides.items():
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    config["train"]["resume"] = False
    path = output_root / "best_configs" / f"{dataset}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train_command(args: argparse.Namespace, experiment: Experiment, trial: Trial, result_dir: Path, checkpoint_dir: Path) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=" + str(args.tune_test_fraction),
        "data.test_split_role=tune",
        "train.resume=false",
        "train.epochs=" + str(args.epochs),
        "train.early_stopping.enabled=true",
        "train.early_stopping.monitor=" + str(args.selection_metric),
        "train.early_stopping.patience=" + str(args.patience),
        "train.early_stopping.min_epochs=" + str(args.min_epochs),
        "graph.use_mask_topology=true",
        "graph.beta_m=1.0",
        "data.root=" + str(experiment.root),
        quoted_override("data.category", experiment.category),
        "data.part_mask_root=" + str(experiment.part_mask_root),
        "train.save_dir=" + str(checkpoint_dir),
        "eval.result_dir=" + str(result_dir),
    ]
    overrides.extend(f"{key}={value}" for key, value in trial.overrides.items())
    return [sys.executable, "scripts/train.py", "--config", args.config, "--set", *overrides]


def ensure_masks_command(args: argparse.Namespace, experiment: Experiment) -> list[str] | None:
    available, total = mask_status(experiment)
    if available >= total:
        return None
    return [
        sys.executable,
        "scripts/generate_part_masks.py",
        "--data-root",
        str(experiment.root),
        "--output-root",
        str(experiment.part_mask_root),
        "--category",
        experiment.category,
        "--device",
        args.sam_device,
    ]


def run(args: argparse.Namespace, command: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


def summarize(args: argparse.Namespace, experiments: list[Experiment], trial_list: list[Trial], base_config: dict[str, Any]) -> None:
    output_root = Path(args.output_root)
    groups = dataset_groups(experiments)
    summary: dict[str, Any] = {}
    for dataset, dataset_experiments in groups.items():
        reps = representative_experiments({dataset: dataset_experiments})
        scores = []
        for trial in trial_list:
            vals = []
            for experiment in reps:
                result = output_root / "results" / dataset / trial.name / experiment.category / f"{experiment.category}_train_metrics.json"
                value = metric_from_result(result, str(args.selection_metric))
                if not math.isnan(value):
                    vals.append(value)
            mean_value = float(sum(vals) / len(vals)) if len(vals) == len(reps) else float("nan")
            scores.append(
                {
                    "trial": trial.name,
                    "mean_selection_metric": mean_value,
                    "selection_metric": str(args.selection_metric),
                    "completed_categories": len(vals),
                    "required_categories": len(reps),
                    "overrides": trial.overrides,
                }
            )
        valid = [item for item in scores if not math.isnan(item["mean_selection_metric"])]
        best = max(valid, key=lambda item: item["mean_selection_metric"]) if valid else None
        best_config = None
        if best is not None:
            best_trial = next(trial for trial in trial_list if trial.name == best["trial"])
            best_config = str(write_best_config(base_config, dataset, best_trial, output_root))
        summary[dataset] = {"scores": scores, "best": best, "best_config": best_config}
    summary_path = output_root / "tuning_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def write_bash(args: argparse.Namespace, experiments: list[Experiment], trial_list: list[Trial]) -> None:
    path = Path(args.write_bash)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}",
        "",
    ]
    reps = representative_experiments(dataset_groups(experiments))
    for experiment in reps:
        mask_command = ensure_masks_command(args, experiment)
        if mask_command is not None:
            lines.append(f"echo '==== SAM {experiment.dataset}/{experiment.category} ===='")
            lines.append(shell_join(mask_command))
        for trial in trial_list:
            result_dir = Path(args.output_root) / "results" / experiment.dataset / trial.name / experiment.category
            checkpoint_dir = Path(args.output_root) / "checkpoints" / experiment.dataset / trial.name / experiment.category
            result_file = result_dir / f"{experiment.category}_train_metrics.json"
            completed_check = shell_join(
                [
                    sys.executable,
                    "-c",
                    "import json,sys; p=sys.argv[1]; data=json.load(open(p, encoding='utf-8')); "
                    "raise SystemExit(0 if data.get('completed') else 1)",
                    str(result_file),
                ]
            )
            lines.append(f"echo '==== tune {experiment.dataset}/{experiment.category}/{trial.name} ===='")
            lines.append(f"if {completed_check} 2>/dev/null; then echo 'SKIP completed'; else")
            lines.append("  " + shell_join(train_command(args, experiment, trial, result_dir, checkpoint_dir)))
            lines.append("fi")
            lines.append("")
    lines.append(shell_join([sys.executable, "scripts/tune_downloaded_datasets.py", "--config", args.config, "--output-root", args.output_root, "--dry-run"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    experiments = downloaded_experiments()
    trial_list = trials()
    if args.write_bash:
        write_bash(args, experiments, trial_list)
        return
    summarize(args, experiments, trial_list, base_config)


if __name__ == "__main__":
    main()
