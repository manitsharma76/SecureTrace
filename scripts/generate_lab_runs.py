#!/usr/bin/env python3
"""Create randomized lab runs for the SecureTrace Docker lab."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SecureTrace lab runs.")
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of randomized lab sessions to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where generated lab artifacts should be saved",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared to generate {args.runs} lab runs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
