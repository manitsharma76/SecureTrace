#!/usr/bin/env python3
"""Phase 3 "before" numbers: evaluate a Phase 1 public-data model on the lab TEST split, using
TRUE lab labels for evaluation only (CLAUDE.md "True lab labels are for evaluation only" -- this
script never trains on the lab, it only scores an already-trained public-data model against it).

Requires a model saved by `scripts/train.py --save-model` (data/processed/models/<experiment>/).

    python scripts/evaluate_lab.py --model-dir data/processed/models/phase1_public_baseline \
        --family xgboost --seed 0 --sessions-dir data/raw/lab/sessions \
        --extracted-dir data/raw/lab/pcap/extracted --held-out-mutator rewrite_size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.data.lab import load_sessions_true_labels
from securetrace.data.lab_split import split_lab_sessions
from securetrace.eval.metrics import evaluate_scores, select_threshold


def load_saved_model(model_dir: Path, family: str, seed: int) -> tuple[object, dict]:
    stem = model_dir / f"{family}_seed{seed}"
    model = joblib.load(f"{stem}.joblib")
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return model, meta


def align_to_training_columns(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reindex the lab features onto the exact training column set and order. A column the public
    model was trained on but that CICFlowMeter didn't produce for the lab (or vice versa) becomes
    NaN, which the model's own imputer (see securetrace/models/factory.py) handles for pipeline
    families; tree models get 0 for missing lab-only columns since XGBoost/LightGBM treat NaN as
    "missing" natively but a column entirely absent must still exist for the right shape."""
    missing = [c for c in feature_columns if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_columns]
    aligned = X.reindex(columns=feature_columns)
    return aligned, missing, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "sessions")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "pcap" / "extracted")
    parser.add_argument("--held-out-mutator", default="rewrite_size",
                        help="Only this mutator's sessions, plus clean sessions, are evaluated here "
                             "(CLAUDE.md: the held-out attack type is used only for testing).")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    manifest_paths = sorted(args.sessions_dir.glob("*.json"))
    if not manifest_paths:
        print(f"No session manifests under {args.sessions_dir}; run scripts/generate_lab_runs.py first.")
        return 1

    split = split_lab_sessions(manifest_paths, held_out_mutator=args.held_out_mutator, seed=args.split_seed)
    if not split.test:
        print("Test split is empty; generate more sessions.")
        return 1
    print(f"Lab test split: {len(split.test)} sessions "
          f"(held-out mutator '{args.held_out_mutator}' + its share of clean/other sessions).")

    X_test, y_test, meta_test = load_sessions_true_labels(split.test, args.extracted_dir)
    print(f"{len(X_test)} lab test flows, {int(y_test.sum())} truly malicious "
          f"({y_test.mean():.1%} attack rate).")

    rows = []
    for seed in args.seeds:
        model, model_meta = load_saved_model(args.model_dir, args.family, seed)
        X_aligned, missing, extra = align_to_training_columns(X_test, model_meta["feature_columns"])
        if seed == args.seeds[0]:
            if missing:
                print(f"WARNING: {len(missing)} training feature(s) not present in lab data (set to NaN): "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
            if extra:
                print(f"{len(extra)} lab feature(s) not used by the model (dropped): "
                      f"{extra[:10]}{'...' if len(extra) > 10 else ''}")

        score = model.predict_proba(X_aligned)[:, 1]
        result = evaluate_scores(y_test.to_numpy().astype(int), score, threshold=model_meta["threshold"],
                                 recalls=(0.90,))
        rows.append({"family": args.family, "held_out_mutator": args.held_out_mutator, "seed": seed, **result})

    results = pd.DataFrame(rows)
    print("\n'Before' numbers (public-data model, no lab adaptation, true lab labels for scoring only):")
    print(results.to_string(index=False))
    if len(results) > 1:
        print(f"\nMean PR-AUC: {results['pr_auc'].mean():.3f} (n={len(results)} seeds)")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Append, replacing earlier rows for the same (family, held-out mutator, seed), so one file
    # accumulates the whole grid of families x held-out choices.
    out_path = out_dir / "phase3_before_numbers.csv"
    if out_path.exists():
        previous = pd.read_csv(out_path)
        if {"family", "held_out_mutator", "seed"} <= set(previous.columns):
            key = ["family", "held_out_mutator", "seed"]
            merged = previous.merge(results[key].drop_duplicates(), on=key, how="left", indicator=True)
            previous = previous[merged["_merge"].eq("left_only").to_numpy()]
            results = pd.concat([previous, results], ignore_index=True)
    results.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
