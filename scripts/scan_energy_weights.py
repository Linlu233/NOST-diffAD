#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from nostdiffad.config import apply_overrides, load_yaml
from nostdiffad.engine import batch_part_masks, load_checkpoint, make_components, make_loader, move_batch
from nostdiffad.metrics import compute_metrics
from nostdiffad.model import NOSTDiffAD
from nostdiffad.utils import resolve_device, seed_everything
from run_downloaded_official_experiments import Experiment, downloaded_experiments, quoted_override
from train import build_datasets


@dataclass(frozen=True)
class WeightCandidate:
    name: str
    alpha: float
    beta: float
    gamma: float
    eta: float
    topk_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="outputs/hparam_tuning/best_configs")
    parser.add_argument("--checkpoint-root", default="outputs/checkpoints_official_tuned")
    parser.add_argument("--output-root", default="outputs/diagnostics/energy_weight_scan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-role", default="tune", choices=["tune", "final", "all"])
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="selection_score")
    parser.add_argument("--selected-summary", default=None)
    parser.add_argument("--write-config-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--au-pro-steps", type=int, default=30)
    parser.add_argument("--upsample-mode", default=None)
    parser.add_argument("--candidate-set", choices=["compact", "full"], default="compact")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def candidates(candidate_set: str) -> list[WeightCandidate]:
    base_weights = [
        ("proto_only", 0.0, 1.0, 0.0, 0.0),
        ("score_only", 1.0, 0.0, 0.0, 0.0),
        ("score_proto_1_1", 1.0, 1.0, 0.0, 0.0),
        ("score_proto_05_1", 0.5, 1.0, 0.0, 0.0),
        ("score_proto_025_1", 0.25, 1.0, 0.0, 0.0),
        ("score_proto_01_1", 0.1, 1.0, 0.0, 0.0),
        ("score_proto_2_1", 2.0, 1.0, 0.0, 0.0),
        ("score_proto_1_2", 1.0, 2.0, 0.0, 0.0),
    ]
    if candidate_set == "compact":
        compact = [
            ("proto_only", 0.0, 1.0, 0.0, 0.0, [0.01, 0.02, 0.05, 0.10, 0.20]),
            ("score_only", 1.0, 0.0, 0.0, 0.0, [0.02, 0.05, 0.10]),
            ("score_proto_01_1", 0.1, 1.0, 0.0, 0.0, [0.02, 0.05, 0.10]),
            ("score_proto_025_1", 0.25, 1.0, 0.0, 0.0, [0.02, 0.05, 0.10]),
            ("score_proto_05_1", 0.5, 1.0, 0.0, 0.0, [0.02, 0.05, 0.10]),
            ("score_proto_1_1", 1.0, 1.0, 0.0, 0.0, [0.02, 0.05, 0.10]),
            ("proto_topo0.0001", 0.0, 1.0, 1e-4, 0.0, [0.05]),
            ("proto_topo0.001", 0.0, 1.0, 1e-3, 0.0, [0.05]),
            ("proto_wave0.0001", 0.0, 1.0, 0.0, 1e-4, [0.05]),
        ]
        return [
            WeightCandidate(f"{name}_topk{topk:g}", alpha, beta, gamma, eta, topk)
            for name, alpha, beta, gamma, eta, topks in compact
            for topk in topks
        ]
    topks = [0.01, 0.02, 0.05, 0.10, 0.20]
    out: list[WeightCandidate] = []
    for name, alpha, beta, gamma, eta in base_weights:
        for topk in topks:
            out.append(WeightCandidate(f"{name}_topk{topk:g}", alpha, beta, gamma, eta, topk))
    for gamma in [1e-4, 1e-3, 1e-2]:
        for topk in [0.02, 0.05, 0.10]:
            out.append(WeightCandidate(f"score_proto_topo{gamma:g}_topk{topk:g}", 1.0, 1.0, gamma, 0.0, topk))
            out.append(WeightCandidate(f"proto_topo{gamma:g}_topk{topk:g}", 0.0, 1.0, gamma, 0.0, topk))
    for eta in [1e-4, 1e-3, 1e-2]:
        for topk in [0.02, 0.05, 0.10]:
            out.append(WeightCandidate(f"score_proto_wave{eta:g}_topk{topk:g}", 1.0, 1.0, 0.0, eta, topk))
            out.append(WeightCandidate(f"proto_wave{eta:g}_topk{topk:g}", 0.0, 1.0, 0.0, eta, topk))
    return out


def metric_value(metrics: dict[str, float], metric: str) -> float:
    value = float(metrics.get(metric, float("nan")))
    return value if math.isfinite(value) else float("nan")


def selected_candidates(path: str | None, all_candidates: list[WeightCandidate]) -> dict[str, WeightCandidate] | None:
    if path is None:
        return None
    by_name = {candidate.name: candidate for candidate in all_candidates}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected: dict[str, WeightCandidate] = {}
    for dataset, item in payload.items():
        best = item.get("best")
        if not best:
            continue
        selected[dataset] = by_name[str(best["candidate"])]
    return selected


def config_path(config_root: Path, experiment: Experiment) -> Path:
    path = config_root / f"{experiment.dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config for {experiment.dataset}: {path}")
    return path


def checkpoint_path(checkpoint_root: Path, experiment: Experiment) -> Path:
    path = checkpoint_root / experiment.dataset / experiment.category / f"{experiment.category}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {experiment.dataset}/{experiment.category}: {path}")
    return path


def experiment_config(args: argparse.Namespace, experiment: Experiment) -> dict[str, Any]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=" + str(args.test_split_fraction),
        "data.test_split_role=" + str(args.split_role),
        "train.resume=false",
        "graph.use_mask_topology=true",
        "graph.beta_m=1.0",
        "energy.alpha=1.0",
        "energy.beta=1.0",
        "energy.gamma=1.0",
        "energy.eta=1.0",
        "data.root=" + str(experiment.root),
        quoted_override("data.category", experiment.category),
        "data.part_mask_root=" + str(experiment.part_mask_root),
    ]
    return apply_overrides(load_yaml(config_path(Path(args.config_root), experiment)), overrides)


@torch.no_grad()
def collect_components(config: dict[str, Any], checkpoint: Path, device: torch.device) -> dict[str, np.ndarray]:
    seed_everything(int(config["seed"]))
    _, val_set, test_set = build_datasets(config, synthetic=False)
    loader = make_loader(test_set, int(config["train"]["batch_size"]), False, int(config["num_workers"]))
    grid = int(config["data"]["image_size"]) // int(config["model"]["patch_size"])
    model = NOSTDiffAD(config).to(device)
    _, proto, nmf, energy = make_components(config, device, grid * grid)
    load_checkpoint(checkpoint, model, proto, nmf, device)
    model.eval()

    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ["score", "proto", "topo", "wave"]}
    grid_hw: tuple[int, int] | None = None
    for batch in tqdm(loader, desc=f"components {config['data']['category']}", leave=False):
        batch = move_batch(batch, device)
        part_masks = batch_part_masks(batch, getattr(model, "require_part_mask", False))
        output = model(batch["image"], part_masks)
        energy_out = energy.total_energy(model, output, batch["category_id"])
        grid_hw = output.grid_hw
        labels.append(batch["label"].detach().cpu().numpy())
        masks.append(batch["mask"][:, 0].detach().cpu().numpy())
        for key in parts:
            parts[key].append(energy_out[key].detach().float().cpu().numpy())
    if grid_hw is None:
        raise RuntimeError("No test batches were collected.")
    _ = val_set
    return {
        "labels": np.concatenate(labels, axis=0),
        "masks": np.concatenate(masks, axis=0),
        "grid_hw": np.asarray(grid_hw, dtype=np.int64),
        **{key: np.concatenate(value, axis=0) for key, value in parts.items()},
    }


def image_scores(patch_energy: np.ndarray, topk_ratio: float) -> np.ndarray:
    k = max(1, int(round(patch_energy.shape[1] * topk_ratio)))
    partition = np.partition(patch_energy, patch_energy.shape[1] - k, axis=1)
    return partition[:, -k:].mean(axis=1)


def score_maps(patch_energy: np.ndarray, grid_hw: tuple[int, int], image_hw: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(patch_energy).float().view(patch_energy.shape[0], 1, grid_hw[0], grid_hw[1])
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        maps = F.interpolate(tensor, size=image_hw, mode=mode, align_corners=False)
    else:
        maps = F.interpolate(tensor, size=image_hw, mode=mode)
    return maps[:, 0].numpy()


def evaluate_candidate(
    components: dict[str, np.ndarray],
    candidate: WeightCandidate,
    au_pro_steps: int,
    upsample_mode: str,
) -> dict[str, float]:
    patch_energy = (
        candidate.alpha * components["score"]
        + candidate.beta * components["proto"]
        + candidate.gamma * components["topo"]
        + candidate.eta * components["wave"]
    )
    masks = components["masks"]
    grid_hw = tuple(int(x) for x in components["grid_hw"])
    maps = score_maps(patch_energy, grid_hw, masks.shape[-2:], upsample_mode)
    return compute_metrics(
        image_labels=components["labels"],
        image_scores=image_scores(patch_energy, candidate.topk_ratio),
        masks=masks,
        score_maps=maps,
        threshold=None,
        inference_seconds=1.0,
        au_pro_steps=au_pro_steps,
    )


def write_best_config(base_config_path: Path, candidate: WeightCandidate, output_root: Path, dataset: str) -> str:
    config = load_yaml(base_config_path)
    config["energy"]["alpha"] = candidate.alpha
    config["energy"]["beta"] = candidate.beta
    config["energy"]["gamma"] = candidate.gamma
    config["energy"]["eta"] = candidate.eta
    config["energy"]["topk_ratio"] = candidate.topk_ratio
    config["train"]["resume"] = False
    config["train"].setdefault("early_stopping", {})
    config["train"]["early_stopping"]["monitor"] = "selection_score"
    path = output_root / f"{dataset}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(path)


def main() -> None:
    args = parse_args()
    all_candidates = candidates(str(args.candidate_set))
    selected = selected_candidates(args.selected_summary, all_candidates)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    rows: list[dict[str, Any]] = []
    experiments = downloaded_experiments()
    if args.dataset:
        experiments = [experiment for experiment in experiments if experiment.dataset == args.dataset]
    if args.category:
        experiments = [experiment for experiment in experiments if experiment.category == args.category]
    if not experiments:
        raise RuntimeError("No matching downloaded experiments found.")
    for experiment in experiments:
        chosen = [selected[experiment.dataset]] if selected and experiment.dataset in selected else all_candidates
        print(
            f"[scan] {experiment.dataset}/{experiment.category} split={args.split_role} candidates={len(chosen)}",
            flush=True,
        )
        config = experiment_config(args, experiment)
        upsample_mode = str(args.upsample_mode or config["energy"].get("upsample_mode", "bilinear"))
        components = collect_components(config, checkpoint_path(Path(args.checkpoint_root), experiment), device)
        category_rows = []
        for candidate in chosen:
            metrics = evaluate_candidate(components, candidate, int(args.au_pro_steps), upsample_mode)
            row = {
                "dataset": experiment.dataset,
                "category": experiment.category,
                "candidate": candidate.name,
                "weights": asdict(candidate),
                "metrics": metrics,
            }
            rows.append(row)
            category_rows.append(row)
        category_path = output_root / "categories" / experiment.dataset / f"{experiment.category}.json"
        category_path.parent.mkdir(parents=True, exist_ok=True)
        category_path.write_text(json.dumps(category_rows, indent=2), encoding="utf-8")

    if args.no_summary:
        return

    summary: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        names = sorted({row["candidate"] for row in dataset_rows})
        scores = []
        for name in names:
            candidate_rows = [row for row in dataset_rows if row["candidate"] == name]
            mean_metrics = {}
            for metric in [
                "selection_score",
                "selection_score_pro",
                "image_auroc",
                "image_ap",
                "pixel_auroc",
                "pixel_ap",
                "au_pro",
                "f1_max",
            ]:
                vals = [metric_value(row["metrics"], metric) for row in candidate_rows]
                vals = [value for value in vals if math.isfinite(value)]
                mean_metrics[metric] = float(sum(vals) / len(vals)) if vals else float("nan")
            scores.append(
                {
                    "candidate": name,
                    "completed_categories": len(candidate_rows),
                    "mean_metrics": mean_metrics,
                    "weights": candidate_rows[0]["weights"],
                }
            )
        valid = [item for item in scores if math.isfinite(float(item["mean_metrics"].get(args.selection_metric, float("nan"))))]
        best = max(valid, key=lambda item: float(item["mean_metrics"][args.selection_metric])) if valid else None
        if best and args.write_config_root:
            candidate = next(item for item in all_candidates if item.name == best["candidate"])
            best["config"] = write_best_config(
                config_path(Path(args.config_root), next(exp for exp in experiments if exp.dataset == dataset)),
                candidate,
                Path(args.write_config_root),
                dataset,
            )
        summary[dataset] = {
            "split_role": args.split_role,
            "selection_metric": args.selection_metric,
            "scores": scores,
            "best": best,
        }

    (output_root / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
