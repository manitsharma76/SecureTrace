#!/usr/bin/env python3
"""Run after a lab session: for each segment's pcap, extract CICFlowMeter features (ML input,
see CLAUDE.md "One flow extractor") and a Zeek conn.log (dashboard only, never a model input).

    python3 lab/sensors/extract.py --pcap-dir /pcap --out-dir /pcap/extracted [--run-id R]

Expects per-segment pcaps named like `<run_id>_seg<A|B|C>.pcap` (as produced by the sensor's
tcpdump capture). Writes, per segment:
    <out-dir>/<run_id>_seg<X>_Flow.csv   (CICFlowMeter; the suffix is the tool's own)
    <out-dir>/<run_id>_seg<X>/conn.log   (Zeek; one subdirectory per segment, Zeek's own layout)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SEGMENT_RE = re.compile(r"^(?P<run_id>.+)_seg(?P<segment>[ABC])\.pcap(?:ng)?$")


def find_segment_pcaps(pcap_dir: Path, run_id: str | None) -> list[tuple[str, str, Path]]:
    """Return (run_id, segment, path) for every matching pcap, optionally filtered to one run."""
    found = []
    for path in sorted(pcap_dir.glob("*.pcap*")):
        match = SEGMENT_RE.match(path.name)
        if not match:
            continue
        if run_id and match["run_id"] != run_id:
            continue
        found.append((match["run_id"], match["segment"], path))
    return found


CFM_BIN_DIR = "/opt/cicflowmeter/bin"  # cfm's generated start script passes the JVM the RELATIVE
# option -Djava.library.path=../lib/native, resolved against the process's cwd, not against the
# script's own location; it must be run from here or the native jnetpcap libraries won't be found.


def run_cicflowmeter(pcap: Path, out_dir: Path) -> Path:
    """cfm writes <pcap-name>_Flow.csv into out_dir; this returns that path once it exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = CFM_BIN_DIR if Path(CFM_BIN_DIR).is_dir() else None
    subprocess.run(["cfm", str(pcap), str(out_dir)], check=True, capture_output=True, text=True, cwd=cwd)
    produced = out_dir / f"{pcap.name}_Flow.csv"
    if not produced.exists():
        raise RuntimeError(f"CICFlowMeter did not produce {produced}")
    return produced


def run_zeek(pcap: Path, out_dir: Path, script: Path) -> Path:
    """Zeek writes into the current directory, so run it with out_dir as cwd."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["zeek", "-C", "-r", str(pcap), str(script)], check=True, cwd=out_dir,
                    capture_output=True, text=True)
    conn_log = out_dir / "conn.log"
    if not conn_log.exists():
        raise RuntimeError(f"Zeek did not produce {conn_log} (no connections in this pcap?)")
    return conn_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcap-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None, help="Only process this run; default: every run found")
    parser.add_argument("--skip-zeek", action="store_true", help="CICFlowMeter only (faster, for ML-only reruns)")
    args = parser.parse_args()

    zeek_script = Path(__file__).with_name("zeek_conn.zeek")
    segments = find_segment_pcaps(args.pcap_dir, args.run_id)
    if not segments:
        print(f"No segment pcaps found under {args.pcap_dir}" + (f" for run {args.run_id}" if args.run_id else ""))
        return 1

    failures = []
    for run_id, segment, pcap in segments:
        label = f"{run_id} segment {segment}"
        try:
            flow_csv = run_cicflowmeter(pcap, args.out_dir)
            print(f"[{label}] CICFlowMeter -> {flow_csv}")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"[{label}] CICFlowMeter FAILED: {exc}", file=sys.stderr)
            failures.append((label, "cicflowmeter", exc))
            continue
        if args.skip_zeek:
            continue
        zeek_out = args.out_dir / f"{run_id}_seg{segment}"
        try:
            conn_log = run_zeek(pcap, zeek_out, zeek_script)
            print(f"[{label}] Zeek -> {conn_log}")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"[{label}] Zeek FAILED (dashboard panel only, not fatal to ML data): {exc}", file=sys.stderr)
            failures.append((label, "zeek", exc))

    if any(tool == "cicflowmeter" for _, tool, _ in failures):
        return 1  # a missing ML extraction is fatal; a missing Zeek log is not
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
