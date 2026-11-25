"""Re-run every lab-dependent experiment, in dependency order, on whatever sessions are currently in
data/raw/lab/sessions. Run this after generating more sessions with generate_lab_runs.py.

    python scripts/run_all.py            # everything
    python scripts/run_all.py --from adapt   # resume from a step

Stops at the first failing step. The Phase 1 public models are NOT retrained (they use public data
only); pass --retrain-public to rebuild them too. Outputs land in data/processed/ and MLflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FAMILIES = ["logreg", "random_forest", "xgboost", "lightgbm", "mlp"]
MUTATORS = ["rewrite_size", "flip_byte"]


def steps(retrain_public: bool) -> list[tuple[str, list[list[str]]]]:
    s: list[tuple[str, list[list[str]]]] = []
    if retrain_public:
        s.append(("public", [[PY, "scripts/train.py", "--config", "configs/phase1_public_baseline.yaml", "--save-model"]]))
    s.append(("gonogo", [[PY, "scripts/go_no_go_check.py"]]))
    s.append(("labelquality", [[PY, "scripts/measure_label_quality.py", "--windows", "0.5", "1", "2", "5", "10"]]))
    s.append(("before", [
        [PY, "scripts/evaluate_lab.py", "--model-dir", "data/processed/models/phase1_public_baseline",
         "--family", fam, "--seeds", "0", "1", "2", "3", "4", "--held-out-mutator", hm]
        for hm in MUTATORS for fam in FAMILIES
    ]))
    s.append(("adapt", [[PY, "scripts/adapt_lab.py"], [PY, "scripts/summarize_phase4.py"]]))
    s.append(("unsup", [[PY, "scripts/compare_unsup_lab.py"], [PY, "scripts/summarize_phase5.py"]]))
    s.append(("fusion", [[PY, "scripts/fuse_lab.py"], [PY, "scripts/summarize_phase6.py"]]))
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", default=None, help="Resume from this step name")
    parser.add_argument("--retrain-public", action="store_true")
    args = parser.parse_args()
    plan = steps(args.retrain_public)
    names = [n for n, _ in plan]
    if args.start and args.start not in names:
        raise SystemExit(f"unknown step {args.start!r}; choose from {names}")
    skipping = args.start is not None
    # evaluate_lab.py appends/replaces rows by (family, mutator, seed), so a clean "before" file avoids stale rows
    for name, commands in plan:
        if skipping and name != args.start:
            continue
        skipping = False
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        for cmd in commands:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"STEP FAILED: {name}: {' '.join(cmd)}", file=sys.stderr)
                return result.returncode
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
