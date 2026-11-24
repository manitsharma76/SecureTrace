#!/usr/bin/env python3
"""Training entry point for SecureTrace experiments (Phase 1: public-data baselines)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.data.public import load_public_frame, split_by_day
from securetrace.eval.metrics import evaluate_scores, select_threshold
from securetrace.models.factory import FAMILIES, build_model, predict_attack_score


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_eval_sets(X, y, meta, data_cfg: dict) -> tuple[dict, dict]:
    """Return (train/val data, named evaluation sets). Split is by day only."""
    train_X, val_X, within_X, train_y, val_y, within_y = split_by_day(
        X,
        y,
        meta["day"],
        train_days=data_cfg["train_days"],
        val_days=data_cfg["val_days"],
        test_days=data_cfg["test_days"],
    )
    cross_days = [str(d) for d in data_cfg.get("cross_test_days", [])]
    used = set(map(str, data_cfg["train_days"])) | set(map(str, data_cfg["val_days"])) | set(map(str, data_cfg["test_days"]))
    if used & set(cross_days):
        raise ValueError("cross_test_days overlap with train/val/test days.")

    eval_sets = {"val": (val_X, val_y), "within_2017": (within_X, within_y)}
    for day in cross_days:
        mask = meta["day"] == day
        eval_sets[f"cross_{day}"] = (X.loc[mask], y.loc[mask])
    if cross_days:
        mask = meta["day"].isin(cross_days)
        eval_sets["cross_all"] = (X.loc[mask], y.loc[mask])
    for name, (features, labels) in {"train": (train_X, train_y), **eval_sets}.items():
        if features.empty:
            raise ValueError(f"Split '{name}' is empty; check the configured days against the data.")
    return {"train": (train_X, train_y), "val": (val_X, val_y)}, eval_sets


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SecureTrace experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_public_baseline.yaml"))
    parser.add_argument("--families", nargs="*", default=None, help="Subset of model families (default: all in config).")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds (smoke runs only).")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    families = args.families or list(config.get("models", {}))
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"Unknown model families: {sorted(unknown)}")
    seeds = args.seeds if args.seeds is not None else config.get("seeds", [config.get("seed", 42)])
    if args.seeds is not None or len(seeds) < 5:
        print(f"WARNING: {len(seeds)} seed(s); reported numbers require >= 5.")

    public_files = [Path(p) for p in data_cfg.get("public_csv_paths", [])]
    missing = [str(p) for p in public_files if not p.exists()]
    if not public_files or missing:
        print(f"Public CSVs missing ({missing or 'none configured'}); run scripts/download_phase1_data.py first.")
        return 1

    print("Loading and cleaning public data ...")
    X, y, meta = load_public_frame(
        public_files,
        label_col=data_cfg.get("label_col", "Label"),
        day_col=data_cfg.get("day_col", "Day"),
        drop_attempted=data_cfg.get("drop_attempted", True),
        drop_features=data_cfg.get("drop_features", []),
    )
    print(f"{len(X):,} flows x {X.shape[1]} shared features")
    print(meta.assign(attack=y).groupby(["dataset", "day"])["attack"].agg(["size", "mean"]).to_string())

    fit_sets, eval_sets = build_eval_sets(X, y, meta, data_cfg)
    train_X, train_y = fit_sets["train"]
    val_X, val_y = fit_sets["val"]

    tracking = config.get("tracking", {})
    uri = tracking.get("mlflow_uri")
    if uri:
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(tracking.get("experiment", "securetrace-phase1"))

    recalls = tuple(config.get("metrics", {}).get("recalls", [0.90, 0.95]))
    max_rows = data_cfg.get("max_train_rows")
    rows: list[dict] = []

    for family in families:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            if max_rows and len(train_X) > max_rows:
                sub = train_X.sample(n=max_rows, random_state=seed).index
                fit_X, fit_y = train_X.loc[sub], train_y.loc[sub]
            else:
                fit_X, fit_y = train_X, train_y

            params = config["models"].get(family, {})
            model = build_model(family, params, seed)
            with mlflow.start_run(run_name=f"{config.get('experiment_name', 'phase1')}-{family}-s{seed}"):
                mlflow.log_params({"family": family, "seed": seed, "n_train": len(fit_X), "n_features": fit_X.shape[1]})
                mlflow.log_params({f"model.{k}": v for k, v in params.items()})
                mlflow.log_param("train_days", ",".join(map(str, data_cfg["train_days"])))
                mlflow.log_artifact(str(args.config))
                model.fit(fit_X, fit_y)

                threshold = select_threshold(val_y.to_numpy(), predict_attack_score(model, val_X))
                mlflow.log_metric("threshold", threshold)
                for name, (features, labels) in eval_sets.items():
                    result = evaluate_scores(
                        labels.to_numpy(), predict_attack_score(model, features), threshold=threshold, recalls=recalls
                    )
                    for metric, value in result.items():
                        if not np.isnan(value):
                            mlflow.log_metric(f"{name}.{metric}", value)
                    rows.append({"family": family, "seed": seed, "eval_set": name, **result})
            last = [r for r in rows if r["family"] == family and r["seed"] == seed and r["eval_set"] == "within_2017"][0]
            print(f"[{family} seed={seed}] within_2017 macro_f1={last['macro_f1']:.3f} pr_auc={last['pr_auc']:.3f}")

    results = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"{config.get('experiment_name', 'phase1')}_results.csv", index=False)

    metric_cols = ["macro_f1", "pr_auc"] + [f"fpr_at_recall_{int(r * 100)}" for r in recalls]
    summary = results.groupby(["family", "eval_set"])[metric_cols].agg(["mean", "std"]).round(3)
    print("\nMean / std over seeds (threshold tuned on the 2017 validation day only):")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
