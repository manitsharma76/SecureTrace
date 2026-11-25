"""Summarise Phase 5 next to the Phase 3 "before" numbers and the Phase 4 weak-label results, as
mean PR-AUC with a 95% t-interval over seeds.

    python scripts/summarize_phase5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def ci(x: pd.Series) -> str:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return f"{x.mean():.3f}" if len(x) else "n/a"
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.3f} +/- {half:.3f}"


def main() -> int:
    p5 = pd.read_csv(P / "phase5_results.csv")
    p5["setting"] = np.where(p5.uses_weak_labels, "weak labels", "label-free")
    p5["name"] = p5["method"]
    p4 = pd.read_csv(P / "phase4_results.csv")
    p4 = p4[(p4.study == "main") & ~p4.shuffled & p4["mode"].isin(["naive", "nnpu", "mil", "elkan_noto"])].copy()
    p4["name"] = "P4 " + p4.family + " " + p4["mode"]
    p4["setting"] = "weak labels"
    before = pd.read_csv(P / "phase3_before_numbers.csv")
    before["name"] = "before: public " + before.family
    before["setting"] = "no adaptation"
    cols = ["held_out_mutator", "seed", "name", "setting", "pr_auc", "fpr_at_recall_90"]
    allr = pd.concat([before[cols], p5[cols], p4[cols]], ignore_index=True)
    out = allr.groupby(["held_out_mutator", "setting", "name"]).agg(
        n_seeds=("seed", "nunique"), pr_auc=("pr_auc", ci), fpr_at_recall_90=("fpr_at_recall_90", "mean"),
    ).reset_index()
    text = "### Phase 5: label-free vs weak-label adaptation (PR-AUC, mean +/- 95% CI over seeds)\n\n"
    text += out.to_markdown(index=False, floatfmt=".3f")

    ssl, scratch = (p5[p5.method == m].set_index(["held_out_mutator", "seed"]).pr_auc
                    for m in ("ssl_weak_finetune", "scratch_weak_finetune"))
    diff = (ssl - scratch).reset_index(name="d")
    text += "\n\n### SSL pretraining contribution (ssl_weak_finetune minus scratch_weak_finetune, paired by seed)\n\n"
    text += diff.groupby("held_out_mutator").d.agg(ci).reset_index().to_markdown(index=False)
    (P / "phase5_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
