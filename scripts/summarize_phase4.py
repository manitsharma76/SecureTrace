"""Summarise Phase 4: paired per-seed PR-AUC gain over the Phase 3 "before" numbers, with 95%
t-intervals over seeds, for the main comparison, the shuffled-label control and the sweeps.

    python scripts/summarize_phase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "processed" / "phase4_results.csv"
BEFORE = ROOT / "data" / "processed" / "phase3_before_numbers.csv"
OUT = ROOT / "data" / "processed" / "phase4_summary.md"


def mean_ci(x: pd.Series) -> str:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return f"{x.mean():.3f}" if len(x) else "n/a"
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.3f} ± {half:.3f}"


def main() -> int:
    res, before = pd.read_csv(RESULTS), pd.read_csv(BEFORE)
    before = before.rename(columns={"pr_auc": "pr_auc_before"})[["family", "held_out_mutator", "seed", "pr_auc_before"]]
    df = res.merge(before, on=["family", "held_out_mutator", "seed"], how="left")
    df["gain"] = df["pr_auc"] - df["pr_auc_before"]
    keys = ["held_out_mutator", "family", "mode", "shuffled", "window_seconds", "tamper_fraction"]
    lines = []

    def table(title: str, frame: pd.DataFrame) -> None:
        agg = frame.groupby(keys).agg(
            n_seeds=("seed", "nunique"), before=("pr_auc_before", "mean"), after=("pr_auc", "mean"),
            gain=("gain", mean_ci),
        ).reset_index()
        lines.append(f"\n### {title}\n")
        lines.append(agg.to_markdown(index=False, floatfmt=".3f"))

    table("Main comparison and shuffled-label control (window 2 s, all tamper sessions)", df[df.study == "main"])
    table("Window-width sensitivity", df[df.study.isin(["window"]) | ((df.study == "main") & ~df.shuffled & df["mode"].isin(["elkan_noto", "nnpu", "mil"]))])
    table("Label efficiency (fraction of tampered training sessions)", df[df.study.isin(["label_efficiency"]) | ((df.study == "main") & ~df.shuffled & df["mode"].isin(["elkan_noto", "nnpu", "mil"]))])

    main_real = df[(df.study == "main") & ~df.shuffled]
    main_shuf = df[(df.study == "main") & df.shuffled]
    ctrl = main_real.merge(main_shuf, on=["held_out_mutator", "seed", "family", "mode"], suffixes=("", "_shuf"))
    ctrl["control_ok"] = ctrl["pr_auc_shuf"] < ctrl["pr_auc"] - 0.05
    lines.append("\n### Shuffled-label control verdict (shuffled PR-AUC must fall clearly below real)\n")
    verdict = ctrl.groupby(["held_out_mutator", "family", "mode"]).agg(
        real=("pr_auc", "mean"), shuffled=("pr_auc_shuf", "mean"),
        seeds_where_gain_vanishes=("control_ok", "sum"), n=("seed", "nunique"),
    ).reset_index()
    lines.append(verdict.to_markdown(index=False, floatfmt=".3f"))

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
