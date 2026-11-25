"""Run the Verification checklist from DEVELOPMENT_PLAN.md and write data/processed/verification_report.md.

    python scripts/verify_project.py

Checks (each can fail): the named integrity/weak-label unit tests, leakage checks (identity columns,
session overlap, held-out attack type, weak labels independent of ground truth), the controls
(shuffled weak labels, segment-only, no cross-segment attribution), reproducibility of a logged
result from its config, and the full test suite. Exit code 1 if anything fails.
End-to-end (docker lab -> dashboard) is scripts/verify_end_to_end.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.adapt.data import build_adapt_data, load_saved_model
from securetrace.data.lab import load_session_manifest
from securetrace.data.lab_split import split_lab_sessions
from securetrace.data.weak_labels import build_session_weak_labels
from securetrace.features.schema import _canonical_name, _is_identity_column

P = ROOT / "data" / "processed"
SESSIONS = ROOT / "data/raw/lab/sessions"
EXTRACTED = ROOT / "data/raw/lab/pcap/extracted"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def pytest_k(expr: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", expr, "tests"], cwd=ROOT, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    return r.returncode == 0, last


def main() -> int:
    manifests = sorted(SESSIONS.glob("*.json"))

    # --- Unit tests named in the plan -----------------------------------------------------------
    for label, expr in [
        ("a one-byte payload flip fails verification", "test_one_byte_payload_flip_fails_verification"),
        ("a tamper on segment k is attributed to k only", "test_tamper_on_segment_k_is_attributed_to_k_only"),
        ("the event log detects a modified row", "test_log_detects_a_modified_row"),
        ("the weak-label builder excludes key/config errors", "test_tamper_events_excludes_config_error_and_incomplete"),
    ]:
        ok, detail = pytest_k(expr)
        check(f"unit: {label}", ok, detail)

    # --- Leakage checks -------------------------------------------------------------------------
    _, meta = load_saved_model(P / "models" / "phase1_public_baseline", "xgboost", 0)
    bad = [c for c in meta["feature_columns"] if _is_identity_column(_canonical_name(c))]
    check("leakage: no identifier/timestamp columns in the model feature set", not bad, f"{len(meta['feature_columns'])} columns, offending: {bad}")

    overlap, heldout_leaks, n_splits = [], [], 0
    for held_out in ("flip_byte", "rewrite_size"):
        for seed in range(5):
            sp = split_lab_sessions(manifests, held_out_mutator=held_out, seed=seed)
            n_splits += 1
            sets = {"train": set(sp.train), "val": set(sp.val), "test": set(sp.test)}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if sets[a] & sets[b]:
                    overlap.append((held_out, seed, a, b))
            for part in ("train", "val"):
                if any(load_session_manifest(p)["plan"]["mutator"] == held_out for p in sets[part]):
                    heldout_leaks.append((held_out, seed, part))
    check("leakage: no session appears in two splits", not overlap, f"{n_splits} splits checked, overlaps: {overlap}")
    check("leakage: the held-out attack type is absent from train and val", not heldout_leaks, f"leaks: {heldout_leaks}")

    with tempfile.TemporaryDirectory() as tmp:
        same = True
        for path in manifests[:12]:
            m = load_session_manifest(path)
            _, y_a, _ = build_session_weak_labels(path, EXTRACTED, window_seconds=2.0)
            stripped = dict(m, ground_truth=[], plan={"tamper_segment": None, "mutator": None})
            alt = Path(tmp) / path.name
            alt.write_text(json.dumps(stripped), encoding="utf-8")
            _, y_b, _ = build_session_weak_labels(alt, EXTRACTED, window_seconds=2.0)
            same &= bool((y_a.to_numpy() == y_b.to_numpy()).all())
    check("leakage: weak labels are identical with ground truth and plan removed from the manifest", same, "12 sessions")

    # --- Controls that must be able to fail -----------------------------------------------------
    p4 = pd.read_csv(P / "phase4_results.csv")
    p4 = p4[p4.study == "main"]
    real, shuf = p4[~p4.shuffled], p4[p4.shuffled]
    j = real.merge(shuf, on=["held_out_mutator", "seed", "family", "mode"], suffixes=("", "_s"))
    arms = j.groupby(["held_out_mutator", "family", "mode"]).apply(
        lambda g: pd.Series({"real": g.pr_auc.mean(), "shuffled": g.pr_auc_s.mean(),
                             "vanish_seeds": int((g.pr_auc_s < g.pr_auc - 0.05).sum()), "n": len(g)}), include_groups=False).reset_index()
    weak_arms = arms[arms["mode"].isin(["naive", "elkan_noto", "mil", "nnpu"])]
    check("control: shuffled weak labels remove the gain (XGBoost/MLP naive, Elkan-Noto, MIL, nnPU)",
          bool((weak_arms.shuffled < weak_arms.real - 0.05).all()),
          f"worst arm real-shuffled gap {np.min(weak_arms.real - weak_arms.shuffled):.3f}; "
          f"seeds where gain vanishes in the weakest arm: {int(weak_arms.vanish_seeds.min())}/{int(weak_arms.n.iloc[0])}")

    gng = subprocess.run([sys.executable, "scripts/go_no_go_check.py"], cwd=ROOT, capture_output=True, text=True).stdout
    auc = float(re.search(r"Mean held-out PR-AUC: ([0-9.]+)", gng).group(1))
    seg = float(re.search(r"segment-identity-only control PR-AUC: ([0-9.]+) \(must be near chance, ~([0-9.]+)\)", gng).group(1))
    chance = float(re.search(r"~([0-9.]+)\)", gng).group(1))
    check("control: go/no-go (true-label model separates attack from benign)", auc >= 0.8, f"PR-AUC {auc:.3f} >= 0.8")
    check("control: a model on segment id alone scores at chance", seg <= chance + 0.05, f"{seg:.3f} vs chance {chance:.3f}")

    mis, tampered, clean_calls = 0, 0, 0
    for path in manifests:
        m = load_session_manifest(path)
        hits = [v["segment"] for v in m["localizations"].values() if v.get("status") == "tamper" and v.get("segment")]
        truth = m["plan"]["tamper_segment"]
        if truth:
            tampered += 1
            mis += sum(1 for s in hits if s != truth)
        else:
            clean_calls += len(hits)
    check("control: no tamper is attributed to the wrong segment; no tamper call on a clean session", mis == 0 and clean_calls == 0,
          f"{tampered} tampered sessions, {mis} misattributed messages, {clean_calls} false calls on clean sessions")

    # --- Reproducibility ------------------------------------------------------------------------
    spec = importlib.util.spec_from_file_location("adapt_lab", ROOT / "scripts" / "adapt_lab.py")
    adapt_lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapt_lab)
    split = split_lab_sessions(manifests, held_out_mutator="flip_byte", seed=0)
    public, pmeta = load_saved_model(P / "models" / "phase1_public_baseline", "xgboost", 0)
    data = build_adapt_data(split, EXTRACTED, pmeta["feature_columns"], window_seconds=2.0, tamper_fraction=1.0, seed=0)
    again = adapt_lab.run_one("xgboost", "naive", data, public, pmeta, 0, False)["pr_auc"]
    logged = float(p4[(p4.held_out_mutator == "flip_byte") & (p4.family == "xgboost") & (p4["mode"] == "naive")
                      & (~p4.shuffled) & (p4.seed == 0)].pr_auc.iloc[0])
    check("reproducibility: re-running a logged Phase 4 cell from its config gives the same PR-AUC",
          abs(again - logged) < 1e-6, f"logged {logged:.6f} vs rerun {again:.6f}")

    # --- Whole suite ----------------------------------------------------------------------------
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    check("full test suite", r.returncode == 0, r.stdout.strip().splitlines()[-1])

    lines = ["# Verification report", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in results]
    (P / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
