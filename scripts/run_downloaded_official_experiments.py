#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from nostdiffad.config import load_yaml
from nostdiffad.data import build_mvtec_records


@dataclass(frozen=True)
class Experiment:
    dataset: str
    root: Path
    category: str
    part_mask_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--best-config-root", default=None)
    parser.add_argument("--output-tag", default="official_tuned")
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-finished", action="store_true")
    parser.add_argument("--write-bash", default=None, help="Write a standalone bash queue and exit.")
    return parser.parse_args()


def categories(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def downloaded_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    mvtec_root = Path("datasets/mvtec_ad")
    if mvtec_root.exists():
        experiments.extend(
            Experiment("mvtec_ad", mvtec_root, category, Path("datasets/part_masks/sam_vit_b_mvtec_ad"))
            for category in categories(mvtec_root)
        )

    btad_root = Path("datasets/btad/BTech_Dataset_transformed")
    if btad_root.exists():
        experiments.extend(
            Experiment("btad", btad_root, category, Path("datasets/part_masks/sam_vit_b_btad"))
            for category in categories(btad_root)
        )

    kolektor_root = Path("datasets/kolektorsdd2_mvtec")
    if kolektor_root.exists():
        experiments.extend(
            Experiment("kolektorsdd2", kolektor_root, category, Path("datasets/part_masks/sam_vit_b_kolektorsdd2"))
            for category in categories(kolektor_root)
        )

    mpdd_root = Path("datasets/MPDD")
    if mpdd_root.exists():
        experiments.extend(
            Experiment("mpdd", mpdd_root, category, Path("datasets/part_masks/sam_vit_b_mpdd"))
            for category in categories(mpdd_root)
        )

    loco_root = Path("datasets/mvtec_loco_anomaly_detection")
    if loco_root.exists():
        experiments.extend(
            Experiment("mvtec_loco", loco_root, category, Path("datasets/part_masks/sam_vit_b_mvtec_loco"))
            for category in categories(loco_root)
        )

    mvtec_ad_2_root = Path("datasets/mvtec_ad_2_mvtec")
    if mvtec_ad_2_root.exists():
        experiments.extend(
            Experiment("mvtec_ad_2", mvtec_ad_2_root, category, Path("datasets/part_masks/sam_vit_b_mvtec_ad_2"))
            for category in categories(mvtec_ad_2_root)
        )

    mvtec_3d_root = Path("datasets/mvtec_3d_rgb_mvtec")
    if mvtec_3d_root.exists():
        experiments.extend(
            Experiment("mvtec_3d_rgb", mvtec_3d_root, category, Path("datasets/part_masks/sam_vit_b_mvtec_3d_rgb"))
            for category in categories(mvtec_3d_root)
        )

    realiad_root = Path("datasets/realiad_1024_mvtec")
    if realiad_root.exists():
        experiments.extend(
            Experiment("realiad_1024", realiad_root, category, Path("datasets/part_masks/sam_vit_b_realiad_1024"))
            for category in categories(realiad_root)
        )

    visa_view_root = Path("datasets/visa_mvtec")
    if visa_view_root.exists():
        experiments.extend(
            Experiment("visa", visa_view_root, category, Path("datasets/part_masks/sam_vit_b_visa"))
            for category in categories(visa_view_root)
        )
    else:
        visa_root = Path("datasets/visa")
        if (visa_root / "VisA_20220922").exists():
            experiments.extend(
                Experiment("visa", visa_root / "VisA_20220922", category, Path("datasets/part_masks/sam_vit_b_visa"))
                for category in categories(visa_root / "VisA_20220922")
            )

    return experiments


def mask_status(experiment: Experiment) -> tuple[int, int]:
    records = build_mvtec_records(
        experiment.root,
        split="train",
        category=experiment.category,
        part_mask_root=experiment.part_mask_root,
    ) + build_mvtec_records(
        experiment.root,
        split="test",
        category=experiment.category,
        part_mask_root=experiment.part_mask_root,
    )
    available = sum(1 for record in records if record.part_mask_path is not None and record.part_mask_path.exists())
    return available, len(records)


def result_path(experiment: Experiment) -> Path:
    return result_path_for_tag(experiment, "official")


def result_path_for_tag(experiment: Experiment, output_tag: str) -> Path:
    return Path("outputs") / f"results_{output_tag}" / experiment.dataset / experiment.category / f"{experiment.category}_train_metrics.json"


def checkpoint_dir_for_tag(experiment: Experiment, output_tag: str) -> Path:
    return Path("outputs") / f"checkpoints_{output_tag}" / experiment.dataset / experiment.category


def result_path_for_args(experiment: Experiment, args: argparse.Namespace) -> Path:
    return result_path_for_tag(experiment, args.output_tag)


def result_file_completed(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("completed")) or len(payload.get("history") or []) >= epochs


def is_finished(experiment: Experiment, epochs: int) -> bool:
    return result_file_completed(result_path(experiment), epochs)


def is_finished_for_args(experiment: Experiment, epochs: int, args: argparse.Namespace) -> bool:
    return result_file_completed(result_path_for_args(experiment, args), epochs)


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def quoted_override(key: str, value: str) -> str:
    return f"{key}='{value}'"


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def config_for_experiment(args: argparse.Namespace, experiment: Experiment) -> str:
    if args.best_config_root is None:
        return args.config
    path = Path(args.best_config_root) / f"{experiment.dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned config for {experiment.dataset}: {path}")
    return str(path)


def train_command(args: argparse.Namespace, experiment: Experiment) -> list[str]:
    return [
        sys.executable,
        "scripts/train.py",
        "--config",
        config_for_experiment(args, experiment),
        "--set",
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=" + str(args.test_split_fraction),
        "data.test_split_role=final",
        "train.resume=true",
        "graph.use_mask_topology=true",
        "graph.beta_m=1.0",
        "data.root=" + str(experiment.root),
        quoted_override("data.category", experiment.category),
        "data.part_mask_root=" + str(experiment.part_mask_root),
        "train.save_dir=" + str(checkpoint_dir_for_tag(experiment, args.output_tag)),
        "eval.result_dir=" + str(Path("outputs") / f"results_{args.output_tag}" / experiment.dataset / experiment.category),
    ]


def write_bash_queue(path: Path, args: argparse.Namespace, experiments: list[Experiment], epochs: int) -> None:
    python_bin = Path(sys.executable)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}",
        f"PY={shlex.quote(str(python_bin))}",
        "",
        "is_finished() {",
        '  "$PY" - "$1" "$2" <<\'PY\'',
        "import json, sys",
        "path, epochs = sys.argv[1], int(sys.argv[2])",
        "try:",
        "    payload = json.load(open(path, 'r', encoding='utf-8'))",
        "except Exception:",
        "    raise SystemExit(1)",
        "raise SystemExit(0 if payload.get('completed') or len(payload.get('history') or []) >= epochs else 1)",
        "PY",
        "}",
        "",
    ]

    for experiment in experiments:
        out_result = result_path_for_args(experiment, args)
        lines.extend(
            [
                f"echo '==== {experiment.dataset}/{experiment.category} ===='",
                f"if is_finished {shlex.quote(str(out_result))} {epochs}; then",
                f"  echo 'SKIP finished {experiment.dataset}/{experiment.category}'",
                "else",
            ]
        )

        available, total = mask_status(experiment)
        if available < total:
            lines.append(f"  echo 'SAM masks {available}/{total}; generating {experiment.dataset}/{experiment.category}'")
            lines.append(
                "  "
                + shell_join(
                    [
                        "$PY",
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
                ).replace(shlex.quote("$PY"), '"$PY"', 1)
            )
        else:
            lines.append(f"  echo 'SAM masks {available}/{total}; ready'")

        lines.append(
            "  "
            + shell_join(train_command(args, experiment)).replace(shlex.quote(sys.executable), '"$PY"', 1)
        )
        lines.append("fi")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    epochs = int(base_config["train"]["epochs"])

    if Path("datasets/kolektorsdd2").exists() and not Path("datasets/kolektorsdd2_mvtec").exists():
        run(
            [
                sys.executable,
                "scripts/prepare_kolektorsdd2_mvtec.py",
                "--source-root",
                "datasets/kolektorsdd2",
                "--output-root",
                "datasets/kolektorsdd2_mvtec",
                "--category",
                "kolektorsdd2",
            ],
            args.dry_run,
        )

    experiments = downloaded_experiments()
    if args.datasets:
        selected = set(args.datasets)
        experiments = [experiment for experiment in experiments if experiment.dataset in selected]
    if not experiments:
        raise RuntimeError("No downloaded supported datasets found under datasets/.")

    if args.write_bash:
        write_bash_queue(Path(args.write_bash), args, experiments, epochs)
        return

    for experiment in experiments:
        if args.skip_finished and is_finished_for_args(experiment, epochs, args):
            print(f"SKIP finished: {experiment.dataset}/{experiment.category}", flush=True)
            continue

        available, total = mask_status(experiment)
        print(
            f"MASK {experiment.dataset}/{experiment.category}: {available}/{total}",
            flush=True,
        )
        if available < total:
            run(
                [
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
                ],
                args.dry_run,
            )
            if args.dry_run:
                continue
            available, total = mask_status(experiment)
            if available < total:
                raise RuntimeError(
                    f"Incomplete SAM part masks for {experiment.dataset}/{experiment.category}: {available}/{total}"
                )

        run(train_command(args, experiment), args.dry_run)


if __name__ == "__main__":
    main()
