"""End-to-end verification (DEVELOPMENT_PLAN.md): docker lab up -> run one tamper session -> the
integrity failure, the localized segment and the ranked fused alert appear in the dashboard.

    python scripts/verify_end_to_end.py [--segment C] [--mutator flip_byte]

Requires the lab to be running (`docker compose --env-file .env -f lab/docker-compose.yml up -d --build`)
and a dashboard bundle (`python scripts/build_dashboard_bundle.py`). The generated session is moved to
data/raw/lab_e2e afterwards so it never leaks into the experiment dataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "raw" / "lab"
E2E = ROOT / "data" / "raw" / "lab_e2e"
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

    before = {p.name for p in (LAB / "sessions").glob("*.json")}
    run = subprocess.run([sys.executable, "scripts/generate_lab_runs.py", "--runs", "1", "--seed", "99",
                          "--force-tamper", args.segment, "--force-mutator", args.mutator], cwd=ROOT, capture_output=True, text=True)
    new = sorted({p.name for p in (LAB / "sessions").glob("*.json")} - before)
    check("one tamper session ran end to end", run.returncode == 0 and len(new) == 1, run.stdout.strip().splitlines()[-3:][0] if run.stdout.strip() else run.stderr[-200:])
    if len(new) != 1:
        return 1
    run_id = new[0].removesuffix(".json")

    import json
    manifest = json.loads((LAB / "sessions" / new[0]).read_text(encoding="utf-8"))
    hits = [v for v in manifest["localizations"].values() if v["status"] == "tamper"]
    check("integrity failure detected", len(hits) > 0, f"{len(hits)} failed messages")
    check("tamper localized to the true segment, none elsewhere",
          hits and all(h["segment"] == args.segment for h in hits), f"expected {args.segment}")
    check("event log hash chain intact", manifest["log_integrity"].get("ok") is True)

    try:
        live = requests.get(f"http://127.0.0.1:9000/runs/{run_id}/localizations", timeout=5).json()
        live_segments = {v["segment"] for v in live.values() if v["status"] == "tamper"}
        check("collector API serves the same localization live", live_segments == {args.segment}, str(live_segments))
    except requests.RequestException as exc:
        check("collector API serves the same localization live", False, str(exc))

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    at.sidebar.checkbox[0].set_value(True).run()
    at.sidebar.selectbox[0].set_value(run_id).run()
    errors = [e.value for e in at.error]
    check("dashboard renders without exceptions", not at.exception, str([e.value for e in at.exception])[:200])
    check("dashboard shows the integrity failure and localized segment",
          any(f"segment {args.segment}" in e for e in errors), "; ".join(errors)[:120])
    tables = [d.value for d in at.dataframe]
    top = next((t for t in tables if "risk" in getattr(t, "columns", [])), None)
    check("dashboard shows the ranked fused alert", top is not None and len(top) > 0 and set(top["segment"]) == {args.segment},
          f"{0 if top is None else len(top)} alert rows")

    E2E.mkdir(parents=True, exist_ok=True)
    for sub in ("sessions", "attacker", "pcap", "pcap/extracted"):
        for f in (LAB / sub).glob(f"*{run_id}*"):
            if f.is_file():
                shutil.move(str(f), str(E2E / f.name))
    print(f"\nEnd-to-end {'PASSED' if OK else 'FAILED'} (session {run_id} archived in {E2E})")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
