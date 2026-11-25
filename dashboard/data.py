"""Cached data access for the dashboard (paths come from environment variables with repo defaults)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.fusion.service import load_bundle, score_session  # noqa: E402
from securetrace.fusion.sessions import SessionRef, catalog, discover  # noqa: E402

BUNDLE = Path(os.environ.get("SECURETRACE_BUNDLE", ROOT / "data/processed/models/dashboard/bundle.joblib"))
SESSIONS = Path(os.environ.get("SECURETRACE_SESSIONS", ROOT / "data/raw/lab/sessions"))
EXTRACTED = Path(os.environ.get("SECURETRACE_EXTRACTED", ROOT / "data/raw/lab/pcap/extracted"))
DEMO_DIR = Path(os.environ.get("SECURETRACE_DEMO_DIR", ROOT / "data/raw/lab_demo"))
COLLECTOR = os.environ.get("SECURETRACE_COLLECTOR", "http://127.0.0.1:9000")
PROCESSED = ROOT / "data" / "processed"
WINDOWS = [0.5, 1.0, 2.0, 5.0, 10.0]


@st.cache_resource(show_spinner=False)
def bundle(path: str = str(BUNDLE)) -> dict:
    return load_bundle(path)


def demo_stamp() -> tuple:
    d = DEMO_DIR / "sessions"
    return tuple(sorted(p.name for p in d.glob("*.json"))) if d.exists() else ()


@st.cache_data(show_spinner=False)
def session_catalog(_stamp: tuple, test_ids: tuple) -> tuple[list[SessionRef], "object"]:
    refs = discover(SESSIONS, EXTRACTED, DEMO_DIR, list(test_ids))
    return refs, catalog(refs)


@st.cache_data(show_spinner=False)
def manifest(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Scoring the session's flows...")
def scored(manifest_path: str, extracted_dir: str, window: float, _bundle: dict):
    """(flows, X) for one session at one weak-label window; cached so sliders stay instant."""
    return score_session(manifest_path, extracted_dir, _bundle, window)
