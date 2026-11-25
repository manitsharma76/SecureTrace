"""Find and describe the lab sessions the dashboard can show (experiment dataset + live-demo runs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from securetrace.fusion.service import session_localization


@dataclass(frozen=True)
class SessionRef:
    run_id: str
    manifest_path: Path
    extracted_dir: Path
    group: str  # "held-out test" | "trained on" | "live demo"


def discover(sessions_dir: Path, extracted_dir: Path, demo_dir: Path, test_run_ids: list[str]) -> list[SessionRef]:
    """Experiment sessions are 'held-out test' if the served model never saw them, else 'trained on'.
    Anything under demo_dir is a 'live demo' run (also unseen by the model)."""
    test = set(test_run_ids)
    refs = [SessionRef(p.stem, p, extracted_dir, "held-out test" if p.stem in test else "trained on")
            for p in sorted(Path(sessions_dir).glob("*.json"))]
    demo_sessions = Path(demo_dir) / "sessions"
    if demo_sessions.exists():
        refs += [SessionRef(p.stem, p, Path(demo_dir) / "extracted", "live demo")
                 for p in sorted(demo_sessions.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True)]
    return refs


def describe(ref: SessionRef) -> dict:
    """Facts for filtering. `integrity_*` come from the observed verdicts; `plan_*` are the attacker's
    own plan, i.e. ground truth, used for navigation and evaluation only."""
    m = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
    loc = session_localization(m)
    return {"run_id": ref.run_id, "group": ref.group,
            "integrity_verdict": "tamper detected" if loc else "no tamper detected",
            "integrity_segment": loc["segment"] if loc else None,
            "plan_segment (evaluation only)": m["plan"].get("tamper_segment"),
            "plan_mutator (evaluation only)": m["plan"].get("mutator")}


def catalog(refs: list[SessionRef]) -> pd.DataFrame:
    return pd.DataFrame([describe(r) for r in refs])
