#!/usr/bin/env python3
"""Download the Phase 1 public datasets into gitignored raw-data storage."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.parse
from pathlib import Path

import requests


ZENODO_CICIDS2017_URL = (
    "https://zenodo.org/api/records/22016274/files/cicids2017.csv/content"
)
ZENODO_CICIDS2017_MD5 = "ea6046dd1a3399ecbc88d682c3b5ba3e"

# Same Zenodo record, same corrected CICFlowMeter: the consistent-extractor cross-dataset target.
ZENODO_INSDN_URL = "https://zenodo.org/api/records/22016274/files/insdn.csv/content"
ZENODO_INSDN_MD5 = "620129018cb4c3c7480bbad07e074d91"

IDS2018_KEYS = (
    "Processed Traffic Data for ML Algorithms/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Processed Traffic Data for ML Algorithms/"
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_ids2017(destination_dir: Path) -> Path:
    destination = destination_dir / "cic_ids_2017_corrected.csv"
    if not destination.exists():
        print(f"Downloading corrected CIC-IDS2017: {destination}")
        download_file(ZENODO_CICIDS2017_URL, destination)
    actual_md5 = md5_file(destination)
    if actual_md5 != ZENODO_CICIDS2017_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {destination}: expected "
            f"{ZENODO_CICIDS2017_MD5}, got {actual_md5}"
        )
    return destination


def download_insdn(destination_dir: Path) -> Path:
    destination = destination_dir / "insdn.csv"
    if not destination.exists():
        print(f"Downloading corrected InSDN: {destination}")
        download_file(ZENODO_INSDN_URL, destination)
    actual_md5 = md5_file(destination)
    if actual_md5 != ZENODO_INSDN_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {destination}: expected {ZENODO_INSDN_MD5}, got {actual_md5}"
        )
    return destination


def download_ids2018_subset(destination_dir: Path) -> list[Path]:
    destinations: list[Path] = []
    for key in IDS2018_KEYS:
        filename = os.path.basename(key)
        destination = destination_dir / "ids2018" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            print(f"Downloading CSE-CIC-IDS2018: {key}")
            url = (
                "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
                + urllib.parse.quote(key)
            )
            download_file(url, destination)
        destinations.append(destination)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Phase 1 public datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/public"),
        help="Gitignored raw-data destination.",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    ids2017 = download_ids2017(args.data_dir)
    insdn = download_insdn(args.data_dir)
    ids2018 = download_ids2018_subset(args.data_dir)
    print(f"InSDN (corrected): {insdn}")
    print(f"CIC-IDS2017: {ids2017}")
    for path in ids2018:
        print(f"CSE-CIC-IDS2018: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
