#!/usr/bin/env python3
"""Phase 3: measure weak-label quality against the TRUE labels, as a function of window width
(DEVELOPMENT_PLAN.md "Measure label quality against the true labels: what fraction of flows in a
positive bag are actually malicious, as a function of window width").

    python scripts/measure_label_quality.py --sessions-dir data/raw/lab/sessions \
        --extracted-dir data/raw/lab/pcap/extracted --windows 0.5 1 2 5 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.data.weak_labels import label_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "sessions")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "pcap" / "extracted")
    parser.add_argument("--windows", nargs="*", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    args = parser.parse_args()

    manifest_paths = sorted(args.sessions_dir.glob("*.json"))
    if not manifest_paths:
        print(f"No session manifests under {args.sessions_dir}; run scripts/generate_lab_runs.py first.")
        return 1

    rows = [label_quality(manifest_paths, args.extracted_dir, window_seconds=w) for w in args.windows]
    results = pd.DataFrame(rows)
    print(f"Label quality across {len(manifest_paths)} sessions, by window width:\n")
    print(results.to_string(index=False))

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "label_quality.csv", index=False)
    print(f"\nSaved: {out_dir / 'label_quality.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
