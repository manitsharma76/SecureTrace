"""Summarise Phase 6: alert quality per mode (mean over incidents, then mean +/- 95% CI over seeds),
false alerts on clean sessions, calibration (ECE) and localization accuracy.

    python scripts/summarize_phase6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P = Path(__file__).resolve().parents[1] / "data" / "processed"


def ci(x: pd.Series) -> str:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return f"{x.mean():.3f}" if len(x) else "n/a"
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.3f} +/- {half:.3f}"


def main() -> int:
    r = pd.read_csv(P / "phase6_results.csv")
    inc = r[(r.kind == "incident") & r.has_true_attack]
    per_seed = inc.groupby(["held_out_mutator", "mode", "seed"]).agg(
        precision_at_5=("precision_at_5", "mean"), precision_at_10=("precision_at_10", "mean"),
        irrelevant_alerts=("irrelevant_alerts", "mean"), alerts=("n_alerts", "mean"),
        recall=("recall", "mean"), incidents=("run_id", "nunique"),
    ).reset_index()
    table = per_seed.groupby(["held_out_mutator", "mode"]).agg(
        incidents=("incidents", "mean"), precision_at_5=("precision_at_5", ci), precision_at_10=("precision_at_10", ci),
        irrelevant_alerts_per_incident=("irrelevant_alerts", ci), alerts_per_incident=("alerts", ci), recall=("recall", ci),
    ).reset_index()
    text = "### Alerting modes on test incidents (sessions with an integrity-localized tamper)\n\n" + table.to_markdown(index=False)

    clean = r[r.kind == "clean_session"]
    cps = clean.groupby(["held_out_mutator", "mode", "seed"]).n_alerts.mean().reset_index()
    ctab = cps.groupby(["held_out_mutator", "mode"]).n_alerts.agg(ci).reset_index().rename(columns={"n_alerts": "false_alerts_per_clean_session"})
    text += "\n\n### False alerts per CLEAN test session\n\n" + ctab.to_markdown(index=False)

    cal = pd.read_csv(P / "phase6_calibration.csv")
    ctab2 = cal.groupby(["held_out_mutator", "calibration"]).ece_vs_true_labels.agg(ci).reset_index()
    text += "\n\n### Calibration: ECE against true labels (lower is better)\n\n" + ctab2.to_markdown(index=False)

    loc = pd.read_csv(P / "phase6_localization.csv")
    text += "\n\n### Localization (all sessions, collector vs attacker plan)\n\n" + loc.to_markdown(index=False)
    (P / "phase6_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
