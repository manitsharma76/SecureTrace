"""Phase 5 (Q3): label-free comparators for adapting to the lab network, scored on the lab TEST
split with TRUE labels (evaluation only), under the same split/seeds as Phase 4.

    python scripts/compare_unsup_lab.py --config configs/phase5_label_free.yaml

Methods use the lab TRAIN sessions' flows WITHOUT labels (anomaly detection, SSL pretraining,
CORAL/DANN target data). Only the two *_weak_finetune arms use the integrity-derived weak labels,
exactly as in Phase 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.adapt.data import build_adapt_data, load_saved_model
from securetrace.data.lab_split import split_lab_sessions
from securetrace.eval.metrics import evaluate_scores
from securetrace.unsup.anomaly import autoencoder_scores, isolation_forest_scores
from securetrace.unsup.da import coral_scores, dann_scores
from securetrace.unsup.prep import load_public_sample, make_prep, shared_columns
from securetrace.unsup.ssl import EMB, embed, make_encoder, scarf_pretrain


def finetune_with_weak_labels(encoder: nn.Sequential, X: np.ndarray, s: np.ndarray, seed: int,
                              epochs: int = 40, lr: float = 1e-3) -> nn.Module:
    """Encoder + linear head trained with the naive weak-label BCE (the Phase 4 winner)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = nn.Sequential(encoder, nn.ReLU(), nn.Linear(EMB, 1))
    Xt, st = torch.tensor(X, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), 128):
            ix = order[i:i + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[ix]).squeeze(-1), st[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def run_method(name: str, data, cols: list[str], X_pub, y_pub, seed: int) -> np.ndarray:
    X_lab, X_test = data.train.X[cols], data.X_test[cols]
    if name == "iforest":
        return isolation_forest_scores(X_lab, X_test, seed)
    if name == "autoencoder":
        return autoencoder_scores(X_lab, X_test, seed)
    if name == "coral":
        return coral_scores(X_pub[cols], y_pub, X_lab, X_test, seed)
    if name == "dann":
        return dann_scores(X_pub[cols], y_pub, X_lab, X_test, seed)

    prep = make_prep().fit(pd.concat([X_pub[cols], X_lab], ignore_index=True))
    L, T = prep.transform(X_lab), prep.transform(X_test)
    if name == "ssl_public_head":
        enc = scarf_pretrain(L, seed)
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=seed).fit(embed(enc, prep.transform(X_pub[cols])), y_pub)
        return clf.predict_proba(embed(enc, T))[:, 1]
    if name in ("ssl_weak_finetune", "scratch_weak_finetune"):
        torch.manual_seed(seed)
        enc = scarf_pretrain(L, seed) if name == "ssl_weak_finetune" else make_encoder(L.shape[1])
        net = finetune_with_weak_labels(enc, L, data.train.s, seed)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(T, dtype=torch.float32)).squeeze(-1)).numpy()
    raise ValueError(f"unknown method {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase5_label_free.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds (smoke runs only)")
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    methods = args.methods or cfg["methods"]
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    manifests = sorted(paths["sessions_dir"].glob("*.json"))
    mlflow.set_tracking_uri(cfg["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    rows = []
    for held_out in cfg["held_out_mutators"]:
        split = split_lab_sessions(manifests, held_out_mutator=held_out, seed=cfg["split_seed"])
        print(f"[{held_out}] train={len(split.train)} val={len(split.val)} test={len(split.test)} sessions")
        for seed in seeds:
            _, meta = load_saved_model(paths["model_dir"], "mlp", seed)
            data = build_adapt_data(split, paths["extracted_dir"], meta["feature_columns"],
                                    window_seconds=cfg["weak_window_seconds"], seed=seed)
            cols = shared_columns(meta["feature_columns"], data.train.X)
            X_pub, y_pub = load_public_sample(paths["public_config"], meta["feature_columns"],
                                              cfg["public_sample_rows"], seed, paths["interim_dir"])
            for name in methods:
                score = run_method(name, data, cols, X_pub, y_pub, seed)
                res = evaluate_scores(data.y_test, score, threshold=0.5, recalls=(0.90,))
                row = {"held_out_mutator": held_out, "seed": seed, "method": name,
                       "uses_weak_labels": name.endswith("weak_finetune"), "n_features": len(cols), **res}
                rows.append(row)
                with mlflow.start_run(run_name=f"{name}-{held_out}-s{seed}"):
                    mlflow.log_params({"method": name, "held_out_mutator": held_out, "seed": seed})
                    mlflow.log_artifact(str(args.config))
                    mlflow.log_metrics({k: v for k, v in res.items() if np.isfinite(v)})
                print(f"  {name:22s} seed={seed} pr_auc={res['pr_auc']:.3f} fpr@r90={res['fpr_at_recall_90']:.3f}")
            pd.DataFrame(rows).to_csv(paths["out_csv"], index=False)
    print(f"Wrote {paths['out_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
