"""End-to-end verification (DEVELOPMENT_PLAN.md): docker lab up -> run one tamper session through the
dashboard's live-demo code path -> the integrity failure, the localized segment and the ranked fused
alert appear in the dashboard.

    python scripts/verify_end_to_end.py [--segment C] [--mutator flip_byte]

Requires the lab to be running (`docker compose --env-file .env -f lab/docker-compose.yml up -d --build`)
and a dashboard bundle (`python scripts/build_dashboard_bundle.py`). The session is archived to
data/raw/lab_demo by the live-demo code, so it never leaks into the experiment dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.fusion.live import run_live_session  # noqa: E402

OK = True


def check(name: str, ok: bool, detail: str = "") -> None:
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", default="C")
    parser.add_argument("--mutator", default="flip_byte")
    args = parser.parse_args()

    up = subprocess.run(["docker", "compose", "--env-file", ".env", "-f", "lab/docker-compose.yml", "ps", "--format",
                         "{{.Service}} {{.State}}"], cwd=ROOT, capture_output=True, text=True).stdout
    check("lab is running", up.count("running") >= 7, up.replace("\n", "; ")[:120])

    steps: list[str] = []
    result = run_live_session(args.segment, args.mutator, steps.append, seed=99)
    check("live demo session ran end to end", result["ok"], (result["error"] or "")[:200])
    if not result["ok"]:
        return 1
    run_id = result["run_id"]
    check("live demo streamed its progress steps", len(steps) >= 6 and any("integrity events" in s for s in steps)
          and any("CICFlowMeter" in s for s in steps), f"{len(steps)} steps")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    hits = [v for v in manifest["localizations"].values() if v["status"] == "tamper"]
    check("integrity failure detected", len(hits) > 0, f"{len(hits)} failed messages")
    check("tamper localized to the true segment, none elsewhere",
          hits and all(h["segment"] == args.segment for h in hits), f"expected {args.segment}")
    check("event log hash chain intact", manifest["log_integrity"].get("ok") is True)
    check("session was archived out of the experiment dataset",
          not (ROOT / "data/raw/lab/sessions" / f"{run_id}.json").exists())

    try:
        live = requests.get(f"http://127.0.0.1:9000/runs/{run_id}/localizations", timeout=5).json()
        live_segments = {v["segment"] for v in live.values() if v["status"] == "tamper"}
        check("collector API serves the same localization live", live_segments == {args.segment}, str(live_segments))
    except requests.RequestException as exc:
        check("collector API serves the same localization live", False, str(exc))

    from streamlit.testing.v1 import AppTest
    os.environ["SECURETRACE_COLLECTOR"] = "http://127.0.0.1:9000"
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=240)
    at.session_state["pending_run"] = run_id  # exactly what the demo's "Open it" button does
    at.run()
    check("dashboard renders without exceptions", not at.exception, str([e.value for e in at.exception])[:200])
    check("dashboard opened the live-demo session", at.sidebar.selectbox(key="run_pick").value == run_id)
    metrics = {m.label: m.value for m in at.metric}
    check("dashboard shows the integrity failure and localized segment",
          metrics.get("3. Tampered segment") == args.segment and "failed" in metrics.get("1. Integrity", ""),
          f"segment {metrics.get('3. Tampered segment')}, {metrics.get('1. Integrity')}")
    next(b for b in at.button if b.key == "card_fused").click().run()
    top = next((d.value for d in at.dataframe if "raw_risk" in getattr(d.value, "columns", [])), None)
    check("dashboard shows the ranked fused alert", top is not None and len(top) > 0 and set(top["segment"]) == {args.segment},
          f"{0 if top is None else len(top)} alert rows")
    check("hop-chain explanation and per-flow SHAP are shown",
          any(h.value.startswith("### Why flow") for h in at.markdown))

    print(f"\nEnd-to-end {'PASSED' if OK else 'FAILED'} (session {run_id} archived in data/raw/lab_demo)")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
