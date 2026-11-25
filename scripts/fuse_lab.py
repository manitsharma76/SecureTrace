"""Phase 6 (Novelty B): localization + calibrated fusion vs ML-only and time-window alerting.

    python scripts/fuse_lab.py --config configs/phase6_fusion.yaml

Per (held-out mutator, seed): adapt the public XGBoost with weak labels (Phase 4 'naive'), fit
isotonic / temperature calibrators on WEAK validation labels, then for every lab TEST session where
the integrity layer localized a tamper (an "incident") build alerts three ways and score them with
the attacker's TRUE labels (evaluation only). Also reports localization accuracy, ECE before/after
calibration, a reliability plot and one example fused alert with TreeSHAP explanations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.adapt.data import build_adapt_data, load_saved_model  # noqa: E402
from securetrace.adapt.tree import adapt_xgboost  # noqa: E402
from securetrace.data.lab import load_session_manifest, load_sessions_true_labels  # noqa: E402
from securetrace.data.lab_split import split_lab_sessions  # noqa: E402
from securetrace.data.weak_labels import build_bags, build_sessions_weak_labels  # noqa: E402
from securetrace.fusion.alerts import candidate_mask, incident_metrics, top_feature_contributions  # noqa: E402
from securetrace.fusion.calibrate import (  # noqa: E402
    IsotonicCalibrator, TemperatureCalibrator, expected_calibration_error, reliability_bins,
)


def session_localization(manifest: dict) -> dict | None:
    """The collector's verdict for the session: the segment named by most 'tamper' messages, with
    one message's integrity evidence attached. None if the integrity layer saw no tamper."""
    hits = [v for v in manifest.get("localizations", {}).values() if v.get("status") == "tamper" and v.get("segment")]
    if not hits:
        return None
    segments = pd.Series([h["segment"] for h in hits])
    seg = segments.mode().iloc[0]
    sample = next(h for h in hits if h["segment"] == seg)
    return {"segment": seg, "n_failed_messages": len(hits), "evidence": sample.get("evidence", {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase6_fusion.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds (smoke runs only)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    thr, window, ks = cfg["alert_threshold"], cfg["window_seconds"], tuple(cfg["ks"])
    manifests = sorted(paths["sessions_dir"].glob("*.json"))
    mlflow.set_tracking_uri(cfg["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    # Localization accuracy needs no model: every session where the collector flagged a tamper.
    loc_rows = []
    for path in manifests:
        m = load_session_manifest(path)
        loc = session_localization(m)
        truth = m["plan"]["tamper_segment"]
        loc_rows.append({"run_id": m["run_id"], "truth": truth, "pred": loc["segment"] if loc else None})
    loc_df = pd.DataFrame(loc_rows)
    tampered = loc_df[loc_df.truth.notna()]
    loc_acc_all = float((tampered.truth == tampered.pred).mean())
    loc_misattributed = int(((tampered.pred.notna()) & (tampered.truth != tampered.pred)).sum())
    clean_false = int(loc_df[loc_df.truth.isna()].pred.notna().sum())
    print(f"Localization over all {len(tampered)} tampered sessions: accuracy {loc_acc_all:.3f}, "
          f"misattributed {loc_misattributed}, false tamper calls on {clean_false} clean sessions")

    rows, calib_rows = [], []
    example_written = False
    for held_out in cfg["held_out_mutators"]:
        split = split_lab_sessions(manifests, held_out_mutator=held_out, seed=cfg["split_seed"])
        for seed in seeds:
            public, meta = load_saved_model(paths["model_dir"], "xgboost", seed)
            cols = meta["feature_columns"]
            data = build_adapt_data(split, paths["extracted_dir"], cols, window_seconds=window, seed=seed)
            model, _ = adapt_xgboost(public, data.train.X, data.train.s, data.train.run_id,
                                     data.val.X, data.val.s, mode="naive", seed=seed)
            raw_val = model.predict_proba(data.val.X)[:, 1]
            raw_test = model.predict_proba(data.X_test)[:, 1]
            iso = IsotonicCalibrator().fit(raw_val, data.val.s)
            temp = TemperatureCalibrator().fit(raw_val, data.val.s)
            variants = {"raw": raw_test, "isotonic": iso.predict(raw_test), "temperature": temp.predict(raw_test)}
            for name, p in variants.items():
                calib_rows.append({"held_out_mutator": held_out, "seed": seed, "calibration": name,
                                   "ece_vs_true_labels": expected_calibration_error(p, data.y_test)})
            if seed == seeds[0]:
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
                for name, p in variants.items():
                    b = reliability_bins(p, data.y_test)
                    ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "o-", label=name)
                ax.set(xlabel="predicted probability", ylabel="observed malicious fraction (true labels)",
                       title=f"Reliability, held-out {held_out}, seed {seed}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(paths["out_dir"] / f"phase6_reliability_{held_out}.png", dpi=130)
                plt.close(fig)

            # Flow metadata for the test sessions, row-aligned with data.X_test (same loaders/order).
            _, y_chk, tmeta = load_sessions_true_labels(split.test, paths["extracted_dir"])
            _, s_flow, wmeta = build_sessions_weak_labels(split.test, paths["extracted_dir"], window_seconds=window)
            assert len(tmeta) == len(wmeta) == len(raw_test)
            assert (tmeta["run_id"].to_numpy() == wmeta["run_id"].to_numpy()).all()
            flows = pd.DataFrame({"run_id": tmeta["run_id"].to_numpy(), "segment": tmeta["segment"].to_numpy(),
                                  "y": data.y_test, "raw": raw_test, "iso": variants["isotonic"],
                                  "in_window": s_flow.to_numpy().astype(bool),
                                  "start": wmeta["flow_start"].to_numpy(), "end": wmeta["flow_end"].to_numpy()})
            for path in split.test:
                m = load_session_manifest(path)
                sess = flows[flows.run_id == m["run_id"]]
                if sess.empty:
                    continue
                loc = session_localization(m)
                is_incident = loc is not None
                # SIEM-style correlation: every flow, on ANY segment, overlapping an integrity-alert
                # time window. Time says when, not where, so other segments' flows are candidates too.
                bags = build_bags(m, window)
                any_seg_window = np.zeros(len(sess), dtype=bool)
                for bag in bags:
                    any_seg_window |= (sess["start"].to_numpy() <= bag.window_end) & (sess["end"].to_numpy() >= bag.window_start)
                for mode, prob_col in (("ml_only", "iso"), ("ml_only_raw_prob", "raw"), ("time_window", "iso"),
                                       ("localized", "iso"), ("localized_raw_prob", "raw")):
                    base = mode.removesuffix("_raw_prob")
                    if base != "ml_only" and not is_incident:
                        mask = np.zeros(len(sess), dtype=bool)  # no integrity failure -> no correlated alert
                    else:
                        win = pd.Series(any_seg_window if base == "time_window" else sess["in_window"].to_numpy())
                        mask = candidate_mask(base, sess, loc, win)
                    res = incident_metrics(sess.y.to_numpy(), sess.raw.to_numpy(), sess[prob_col].to_numpy(),
                                           mask, thr, ks)
                    rows.append({"held_out_mutator": held_out, "seed": seed, "run_id": m["run_id"], "mode": mode,
                                 "kind": "incident" if is_incident else "clean_session",
                                 "has_true_attack": bool(sess.y.sum() > 0), **res})
                if is_incident and not example_written:
                    cand = sess[candidate_mask("localized", sess, loc, sess["in_window"])]
                    top = cand.sort_values("raw", ascending=False).head(3)
                    feats = top_feature_contributions(model.get_booster(), data.X_test.loc[top.index])
                    example = {"run_id": m["run_id"], "localized_segment": loc["segment"],
                               "integrity_evidence": loc["evidence"], "n_failed_messages": loc["n_failed_messages"],
                               "alerts": [{"flow_row": int(i), "calibrated_probability": float(r.iso),
                                           "true_malicious": int(r.y),
                                           "top_features_treeshap": f} for (i, r), f in zip(top.iterrows(), feats)]}
                    (paths["out_dir"] / "phase6_example_alert.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
                    example_written = True
            print(f"[{held_out}] seed {seed} done")
            with mlflow.start_run(run_name=f"fusion-{held_out}-s{seed}"):
                mlflow.log_params({"held_out_mutator": held_out, "seed": seed, "window_seconds": window})
                mlflow.log_artifact(str(args.config))
                mlflow.log_metrics({f"ece_{c['calibration']}": c["ece_vs_true_labels"]
                                    for c in calib_rows if c["seed"] == seed and c["held_out_mutator"] == held_out})

    pd.DataFrame(rows).to_csv(paths["out_dir"] / "phase6_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(paths["out_dir"] / "phase6_calibration.csv", index=False)
    pd.DataFrame([{"n_tampered_sessions": len(tampered), "localization_accuracy": loc_acc_all,
                   "misattributed": loc_misattributed, "false_tamper_calls_on_clean": clean_false}]
                 ).to_csv(paths["out_dir"] / "phase6_localization.csv", index=False)
    print("Wrote phase6_results.csv / phase6_calibration.csv / phase6_localization.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
