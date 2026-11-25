"""Train the model the dashboard serves and save it as one bundle (model + isotonic calibrator +
feature columns + the list of held-out TEST session ids the dashboard offers by default).

    python scripts/build_dashboard_bundle.py --held-out-mutator flip_byte --seed 0

Uses only weak labels for training/calibration (Phase 4 'naive' XGBoost adaptation + Phase 6
isotonic calibrator), exactly as scripts/fuse_lab.py does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.adapt.data import build_adapt_data, load_saved_model
from securetrace.adapt.tree import adapt_xgboost
from securetrace.data.lab import load_session_manifest
from securetrace.data.lab_split import split_lab_sessions
from securetrace.fusion.calibrate import IsotonicCalibrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out-mutator", default="flip_byte")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/processed/models/phase1_public_baseline")
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "data/raw/lab/sessions")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "data/raw/lab/pcap/extracted")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/models/dashboard/bundle.joblib")
    args = parser.parse_args()

    manifests = sorted(args.sessions_dir.glob("*.json"))
    split = split_lab_sessions(manifests, held_out_mutator=args.held_out_mutator, seed=0)
    public, meta = load_saved_model(args.model_dir, "xgboost", args.seed)
    data = build_adapt_data(split, args.extracted_dir, meta["feature_columns"],
                            window_seconds=args.window_seconds, seed=args.seed)
    model, _ = adapt_xgboost(public, data.train.X, data.train.s, data.train.run_id,
                             data.val.X, data.val.s, mode="naive", seed=args.seed)
    calibrator = IsotonicCalibrator().fit(model.predict_proba(data.val.X)[:, 1], data.val.s)
    test_ids = [load_session_manifest(p)["run_id"] for p in split.test]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "columns": meta["feature_columns"],
                 "test_run_ids": test_ids, "window_seconds": args.window_seconds,
                 "held_out_mutator": args.held_out_mutator, "seed": args.seed}, args.out)
    print(f"Saved {args.out}  ({len(test_ids)} held-out test sessions offered by default)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
