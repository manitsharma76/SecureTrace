#!/usr/bin/env python3
"""Phase 2 go/no-go check (DEVELOPMENT_PLAN.md): "a model trained on the TRUE lab labels must
separate attack from benign flows. If it cannot, the attacks need more footprint before any
weak-label work starts."

This is the only place in the project that trains directly on true lab labels; everywhere else
(Phase 4+) uses only the weak labels from integrity failures, per CLAUDE.md "True lab labels are
for evaluation only." This script exists purely to answer one yes/no question before investing in
the weak-label pipeline, not to produce a reportable model.

    python scripts/go_no_go_check.py --sessions-dir data/raw/lab/sessions \
        --extracted-dir data/raw/lab/pcap/extracted

Splits by SESSION (never by row, per CLAUDE.md), trains one XGBoost model per seed on all but one
held-out session, and reports PR-AUC / FPR-at-recall on the held-out session(s). Also runs the
localization control: a model trained on segment identity alone must score at chance, confirming
the signal comes from flow behaviour, not from which bridge a flow happened to be captured on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.data.lab import load_session_manifest, load_sessions_true_labels
from securetrace.eval.metrics import evaluate_scores, select_threshold
from securetrace.models.factory import build_model, predict_attack_score

SEEDS = (0, 1, 2, 3, 4)


def session_splits(
    manifest_paths: list[Path], seed: int, test_fraction: float = 0.3
) -> tuple[list[Path], list[Path]]:
    """Split whole sessions into train/test (never rows). Tampered and clean sessions are
    shuffled and held out separately so the test fold is guaranteed to contain both -- otherwise
    PR-AUC is undefined about a third of the time by chance alone with this few sessions."""
    rng = np.random.RandomState(seed)

    def split_one(paths: list[Path]) -> tuple[list[Path], list[Path]]:
        if not paths:
            return [], []
        order = rng.permutation(len(paths))
        shuffled = [paths[i] for i in order]
        n_test = max(1, round(len(shuffled) * test_fraction))
        return shuffled[n_test:], shuffled[:n_test]

    tampered = [p for p in manifest_paths if load_session_manifest(p)["plan"]["tamper_segment"]]
    clean = [p for p in manifest_paths if not load_session_manifest(p)["plan"]["tamper_segment"]]
    train_t, test_t = split_one(tampered)
    train_c, test_c = split_one(clean)
    return train_t + train_c, test_t + test_c


def run_for_seed(manifest_paths: list[Path], extracted_dir: Path, seed: int) -> dict:
    train_paths, test_paths = session_splits(manifest_paths, seed)
    X_train, y_train, meta_train = load_sessions_true_labels(train_paths, extracted_dir)
    X_test, y_test, meta_test = load_sessions_true_labels(test_paths, extracted_dir)
    shared = sorted(set(X_train.columns) & set(X_test.columns))
    X_train, X_test = X_train[shared], X_test[shared]

    if y_train.nunique() < 2:
        raise ValueError(
            f"Seed {seed}: training sessions {[p.stem for p in train_paths]} contain only one "
            "class; regenerate more sessions (need both tamper and clean) before this check means anything."
        )

    model = build_model("xgboost", {"n_estimators": 150, "max_depth": 4}, seed)
    model.fit(X_train, y_train.astype(int))
    val_threshold = select_threshold(y_train.to_numpy().astype(int), predict_attack_score(model, X_train))
    score = predict_attack_score(model, X_test)
    result = evaluate_scores(y_test.to_numpy().astype(int), score, threshold=val_threshold, recalls=(0.90,))

    # Control: segment identity alone must NOT predict tampering (it would mean the model is
    # learning "segment B is always the bad one" instead of flow behaviour -- see
    # DEVELOPMENT_PLAN.md "biggest risk"). This is deliberately leakage the real pipeline forbids
    # (CLAUDE.md "no identity features"); it exists only as a control, never as a real model.
    segment_only_train = pd.get_dummies(meta_train["segment"], prefix="segment")
    segment_only_test = pd.get_dummies(meta_test["segment"], prefix="segment").reindex(
        columns=segment_only_train.columns, fill_value=0
    )
    control_model = build_model("logreg", {}, seed)
    control_model.fit(segment_only_train, y_train.astype(int))
    control_score = predict_attack_score(control_model, segment_only_test)
    control_result = evaluate_scores(y_test.to_numpy().astype(int), control_score, threshold=0.5, recalls=())

    return {
        "seed": seed, "test_sessions": [p.stem for p in test_paths], **result,
        "segment_only_pr_auc": control_result["pr_auc"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "sessions")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "data" / "raw" / "lab" / "pcap" / "extracted")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--pr-auc-threshold", type=float, default=0.8,
                        help="Minimum mean held-out PR-AUC to call this a GO")
    args = parser.parse_args()

    manifest_paths = sorted(args.sessions_dir.glob("*.json"))
    if len(manifest_paths) < 3:
        print(f"Only {len(manifest_paths)} session(s) under {args.sessions_dir}; "
              "run scripts/generate_lab_runs.py --runs <N> first (need several, with both clean and tampered).")
        return 1

    tampered_sessions = sum(1 for p in manifest_paths if load_session_manifest(p)["plan"]["tamper_segment"])
    print(f"{len(manifest_paths)} sessions found ({tampered_sessions} tampered, "
          f"{len(manifest_paths) - tampered_sessions} clean).")
    if tampered_sessions == 0 or tampered_sessions == len(manifest_paths):
        print("Need both tampered and clean sessions for a meaningful check.")
        return 1

    rows = []
    for seed in args.seeds:
        try:
            rows.append(run_for_seed(manifest_paths, args.extracted_dir, seed))
        except ValueError as exc:
            print(f"  seed {seed}: SKIPPED ({exc})")
    if not rows:
        print("No seed produced a usable split; add more sessions.")
        return 1

    results = pd.DataFrame(rows)
    print("\nPer-seed results (true-label model, held-out session(s)):")
    print(results[["seed", "test_sessions", "n", "attack_rate", "pr_auc", "fpr_at_recall_90",
                   "segment_only_pr_auc"]].to_string(index=False))

    mean_pr_auc = results["pr_auc"].mean()
    mean_segment_only = results["segment_only_pr_auc"].mean()
    print(f"\nMean held-out PR-AUC: {mean_pr_auc:.3f} (threshold for GO: {args.pr_auc_threshold})")
    print(f"Mean segment-identity-only control PR-AUC: {mean_segment_only:.3f} (must be near chance, ~{results['attack_rate'].mean():.3f})")

    go = mean_pr_auc >= args.pr_auc_threshold
    control_clean = mean_segment_only < mean_pr_auc - 0.1  # must not already explain the real result
    print(f"\nVerdict: {'GO' if go else 'NO-GO'}"
          + ("" if control_clean else " (but the segment-identity control is suspiciously high -- "
             "check the attack footprint before trusting a GO)"))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
