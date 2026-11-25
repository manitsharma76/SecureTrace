"""Phase 4 (Novelty A): adapt public-data models with integrity-derived WEAK labels and score them
on the lab TEST split with TRUE labels (evaluation only, never training).

    python scripts/adapt_lab.py --config configs/phase4_adaptation.yaml

For every (held-out mutator, seed) the public model of that seed is adapted by each method in the
config, and also by the same method trained on SHUFFLED weak labels -- the control that must
remove the gain (CLAUDE.md). Rows go to the CSV in the config and to MLflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.adapt.data import build_adapt_data, load_saved_model
from securetrace.adapt.neural import from_public_pipeline, fit_neural, score_neural
from securetrace.adapt.pu import (
    estimate_class_prior, estimate_label_frequency, fit_labelled_vs_unlabelled,
)
from securetrace.adapt.tree import adapt_xgboost
from securetrace.data.lab_split import split_lab_sessions
from securetrace.eval.metrics import evaluate_scores, select_threshold


def _shuffled(s: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(10_000 + seed).permutation(s)


def run_one(family: str, mode: str, data, public, meta: dict, seed: int, shuffle: bool) -> dict:
    s_train = _shuffled(data.train.s, seed) if shuffle else data.train.s
    s_val = _shuffled(data.val.s, seed) if shuffle else data.val.s
    info: dict = {}
    if family == "xgboost":
        model, info = adapt_xgboost(public, data.train.X, s_train, data.train.run_id,
                                    data.val.X, s_val, mode=mode, seed=seed)
        score_val = model.predict_proba(data.val.X)[:, 1]
        score_test = model.predict_proba(data.X_test)[:, 1]
    else:
        g = fit_labelled_vs_unlabelled(data.train.X, s_train, seed)
        c = estimate_label_frequency(g, data.val.X, s_val)
        # "_lf" variants: class prior = labelled fraction (assumes labels are complete, c = 1)
        # instead of the Elkan-Noto estimate, which assumes SCAR and overshoots when the integrity
        # bags capture nearly every true positive. Reported as an ablation, not tuned on test.
        base_mode = mode.removesuffix("_lf")
        c_used = 1.0 if mode.endswith("_lf") else c
        if base_mode == "mil":
            bag_pos = pd.Series(s_train).groupby(data.train.bag_id).max()
            prior = float(np.clip(bag_pos.mean() / c_used, 0.01, 0.9))
        else:
            prior = estimate_class_prior(s_train, c_used)
        info.update(label_frequency_c=c, class_prior=prior)
        mode = base_mode
        preprocess, net = from_public_pipeline(public)
        net = fit_neural(preprocess, net, data.train.X, s_train, data.train.bag_id,
                         mode=mode, prior=prior, seed=seed)
        score_val = score_neural(preprocess, net, data.val.X)
        score_test = score_neural(preprocess, net, data.X_test)
    # Threshold from the weak validation labels only (true labels never touch a threshold).
    threshold = select_threshold(s_val, score_val) if len(np.unique(s_val)) == 2 else 0.5
    metrics = evaluate_scores(data.y_test, score_test, threshold=threshold, recalls=(0.90,))
    return {**metrics, **info}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4_adaptation.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds (smoke runs only)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    manifests = sorted(paths["sessions_dir"].glob("*.json"))
    mlflow.set_tracking_uri(cfg["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    main_cfg, sweep = cfg["main"], cfg["sweeps"]
    plan = []  # (window, fraction, [(family, mode)], shuffle_too, tag)
    plan.append((main_cfg["window_seconds"], main_cfg["tamper_fraction"], main_cfg["methods"], True, "main"))
    for w in sweep["window_seconds"]:
        plan.append((w, main_cfg["tamper_fraction"], sweep["methods"], False, "window"))
    for f in sweep["tamper_fraction"]:
        plan.append((main_cfg["window_seconds"], f, sweep["methods"], False, "label_efficiency"))

    rows = []
    for held_out in cfg["held_out_mutators"]:
        split = split_lab_sessions(manifests, held_out_mutator=held_out, seed=cfg["split_seed"])
        print(f"[{held_out}] train={len(split.train)} val={len(split.val)} test={len(split.test)} sessions")
        for seed in seeds:
            for window, fraction, methods, shuffle_too, tag in plan:
                for family, mode in methods:
                    public, meta = load_saved_model(paths["model_dir"], family, seed)
                    data = build_adapt_data(split, paths["extracted_dir"], meta["feature_columns"],
                                            window_seconds=window, tamper_fraction=fraction, seed=seed)
                    for shuffle in ([False, True] if shuffle_too else [False]):
                        res = run_one(family, mode, data, public, meta, seed, shuffle)
                        row = {"held_out_mutator": held_out, "seed": seed, "family": family, "mode": mode,
                               "shuffled": shuffle, "window_seconds": window, "tamper_fraction": fraction,
                               "study": tag, "n_train": len(data.train.s), "n_weak_pos": int(data.train.s.sum()),
                               **res}
                        rows.append(row)
                        with mlflow.start_run(run_name=f"{family}-{mode}-{'shuf-' if shuffle else ''}{held_out}-s{seed}-{tag}"):
                            mlflow.log_params({k: row[k] for k in ("held_out_mutator", "seed", "family", "mode", "shuffled",
                                                                   "window_seconds", "tamper_fraction", "study")})
                            mlflow.log_artifact(str(args.config))
                            mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, float) and np.isfinite(v)})
                        print(f"  {tag:16s} {family:8s} {mode:10s} shuf={shuffle!s:5s} w={window:<4} f={fraction:<4} "
                              f"seed={seed} pr_auc={res['pr_auc']:.3f}")
            pd.DataFrame(rows).to_csv(paths["out_csv"], index=False)
    print(f"Wrote {paths['out_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
