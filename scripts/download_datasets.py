#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    url: str | None
    archive_name: str | None
    target_dir: str
    manual_reason: str | None = None

    @property
    def automatic(self) -> bool:
        return self.url is not None and self.archive_name is not None


DATASETS: dict[str, DatasetSpec] = {
    "mvtec_ad": DatasetSpec(
        "mvtec_ad",
        "MVTec AD",
        "https://www.mydrive.ch/shares/38536/9977512462b47b22694b31d3f4dd1b11910e819d6e5349a39f994f8b8601a776/mvtec_anomaly_detection.tar.xz",
        "mvtec_anomaly_detection.tar.xz",
        "mvtec_ad",
    ),
    "visa": DatasetSpec(
        "visa",
        "VisA",
        "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "VisA_20220922.tar",
        "visa",
    ),
    "btad": DatasetSpec(
        "btad",
        "BTAD",
        "https://avires.dimi.uniud.it/papers/btad/btad.zip",
        "btad.zip",
        "btad",
    ),
    "kolektorsdd2": DatasetSpec(
        "kolektorsdd2",
        "KolektorSDD2",
        "https://go.vicos.si/kolektorsdd2",
        "KolektorSDD2.zip",
        "kolektorsdd2",
    ),
    "mvtec_ad_2": DatasetSpec(
        "mvtec_ad_2",
        "MVTec AD 2",
        None,
        None,
        "mvtec_ad_2",
        "Official release requires the MVTec evaluation server / access flow; no stable public archive URL is exposed for unattended download.",
    ),
    "mvtec_loco_ad": DatasetSpec(
        "mvtec_loco_ad",
        "MVTec LOCO AD",
        None,
        None,
        "mvtec_loco_ad",
        "Official MVTec page does not expose a stable direct full-data archive URL for unattended download in this environment.",
    ),
    "mvtec_3d_ad": DatasetSpec(
        "mvtec_3d_ad",
        "MVTec 3D-AD",
        None,
        None,
        "mvtec_3d_ad",
        "Official MVTec page does not expose a stable direct full-data archive URL for unattended download in this environment.",
    ),
    "mpdd": DatasetSpec(
        "mpdd",
        "MPDD",
        None,
        None,
        "mpdd",
        "Original distribution is behind the provider's browser/session based share, so unattended download needs a user-provided URL.",
    ),
    "real_iad": DatasetSpec(
        "real_iad",
        "Real-IAD",
        None,
        None,
        "real_iad",
        "Hugging Face gated access requires a logged-in account/token and accepted terms.",
    ),
    "dagm": DatasetSpec(
        "dagm",
        "DAGM",
        None,
        None,
        "dagm",
        "Original DAGM 2007 access is form/session based; unattended download needs a user-provided official archive URL.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["visa", "btad", "kolektorsdd2"],
        help=f"Dataset keys. Use 'all-auto' for all automatic public sources. Choices: {', '.join(DATASETS)}",
    )
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def selected_specs(keys: list[str]) -> list[DatasetSpec]:
    if keys == ["all-auto"]:
        return [spec for spec in DATASETS.values() if spec.automatic]
    specs = []
    for key in keys:
        normalized = key.lower().replace("-", "_")
        if normalized not in DATASETS:
            raise SystemExit(f"Unknown dataset key: {key}")
        specs.append(DATASETS[normalized])
    return specs


def has_extracted_content(target_dir: Path, archive_path: Path) -> bool:
    if not target_dir.exists():
        return False
    return any(path != archive_path for path in target_dir.iterdir())


def download(url: str, output: Path, force: bool) -> None:
    aria2_state = output.with_suffix(output.suffix + ".aria2")
    if output.exists() and output.stat().st_size > 0 and not aria2_state.exists() and not force:
        print(f"Using existing archive: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".part")
    if shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "-c",
            "-x16",
            "-s16",
            "-k1M",
            "--summary-interval=30",
            "--console-log-level=notice",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--dir",
            str(output.parent),
            "--out",
            output.name,
            url,
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return
    if shutil.which("wget"):
        cmd = ["wget", "-c", "--progress=dot:giga", "-O", str(tmp_path), url]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        tmp_path.replace(output)
        return
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--continue-at", "-", "--output", str(tmp_path), url]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        tmp_path.replace(output)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "NOST-DiffAD dataset downloader"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        with tmp_path.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=output.name) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bar.update(len(chunk))
    tmp_path.replace(output)


def extract_archive(archive_path: Path, target_dir: Path, force: bool) -> None:
    if has_extracted_content(target_dir, archive_path) and not force:
        print(f"Using existing extracted directory: {target_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {target_dir}")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(target_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path}")


def main() -> None:
    args = parse_args()
    if args.list:
        for spec in DATASETS.values():
            status = "auto" if spec.automatic else "manual"
            print(f"{spec.key}: {spec.display_name} [{status}]")
            if spec.manual_reason:
                print(f"  {spec.manual_reason}")
        return

    root = Path(args.root)
    for spec in selected_specs(args.datasets):
        if not spec.automatic:
            print(f"Skipping {spec.display_name}: {spec.manual_reason}")
            continue
        target_dir = root / spec.target_dir
        archive_path = target_dir / str(spec.archive_name)
        print(f"Fetching {spec.display_name} from official/original source:")
        print(f"  {spec.url}")
        download(str(spec.url), archive_path, bool(args.force))
        if not args.no_extract:
            extract_archive(archive_path, target_dir, bool(args.force))
        if not args.keep_archive and spec.key != "mvtec_ad":
            try:
                archive_path.unlink()
            except FileNotFoundError:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
