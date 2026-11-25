#!/usr/bin/env python3
"""Run N randomized SecureTrace lab sessions end to end and write pcaps, CICFlowMeter CSVs,
Zeek conn.logs, integrity events/localization, and true per-message labels.

Prerequisites: the lab is up (`docker compose --env-file .env -f lab/docker-compose.yml up -d
--build`); NIC offloads are disabled automatically at the start of main().

    python scripts/generate_lab_runs.py --runs 20 --out-dir data/raw/lab

Each session:
  1. Picks a tamper segment (A/B/C, or none for a clean session) and, if tampering, a mutator
     from lab/attacker/proxy.py and whether a recon step precedes it (DEVELOPMENT_PLAN.md
     "biggest risk": tamper placement, timing and benign mix are randomized per run).
  2. Starts the attacker relay on EVERY segment (pass-through on the two not targeted this
     session -- see lab/attacker/proxy.py's module docstring for why this is required even when
     nothing is being tampered with), starts tcpdump capture, starts background noise, runs the
     sender for --n messages, then tears everything down.
  3. Runs lab/sensors/extract.py (CICFlowMeter + Zeek) on the captured pcaps.
  4. Reads the collector's per-message localization (the weak-label source) and the attacker's
     own ground-truth JSONL (the TRUE label, used only for evaluation -- never for training,
     per CLAUDE.md "True lab labels are for evaluation only").
  5. Writes one run manifest (JSON) tying all of this together.

This intentionally drives the real `docker compose exec` containers rather than reimplementing
any of their logic, so a session here exercises exactly what a human operator would.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.attacker.proxy import MUTATORS  # noqa: E402  (after sys.path fixup)
from securetrace.integrity.chain import SEGMENT_ORDER  # noqa: E402

COMPOSE_FILE = ROOT / "lab" / "docker-compose.yml"
ENV_FILE = ROOT / ".env"
COLLECTOR_URL = "http://127.0.0.1:9000"

# lab/docker-compose.yml's fixed addresses (see that file's networks: block).
SEGMENT_IN_ADDR = {"A": "172.28.11.66", "B": "172.28.21.66", "C": "172.28.31.66"}
SEGMENT_OUT_ADDR = {"A": "172.28.12.66", "B": "172.28.22.66", "C": "172.28.32.66"}
DOWNSTREAM_ADDR = {"A": "172.28.12.12", "B": "172.28.22.12", "C": "172.28.32.12"}  # relay1/relay2/receiver


def compose_exec(service: str, *args: str, detach: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "--env-file", str(ENV_FILE), "-f", str(COMPOSE_FILE), "exec"]
    if detach:
        cmd.append("-d")
    cmd += [service, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def disable_offloads() -> None:
    """Turn off TSO/GSO/GRO/LRO on every interface the sensor (host network) can see, so pcaps hold
    real wire-size packets rather than coalesced super-frames. Idempotent; per-interface failures
    (virtual devices without the feature) are ignored."""
    script = ("for i in $(ls /sys/class/net); do "
              "ethtool -K $i tso off gso off gro off lro off >/dev/null 2>&1; done; true")
    compose_exec("sensor", "sh", "-c", script, check=False)


def start_benign_servers() -> None:
    """Listeners for the benign noise clients on the attacker node. Noise has to cross the same
    point-to-point segment networks the transfer traffic does (the sender can only reach the
    attacker's in-side address), otherwise it is never captured and every lab flow is a transfer
    flow, which makes weak-label bags trivially pure. The relays/receiver already run their own
    listeners. Idempotent: a second listener fails to bind and exits."""
    compose_exec_bg("attacker", "python", "-m", "lab.agents.benign_servers")
    time.sleep(1.0)


def compose_exec_bg(service: str, *args: str) -> None:
    """Fire-and-forget inside the container (the attacker relays, noise generators)."""
    compose_exec(service, *args, detach=True)


def kill_background(service: str, pattern: str) -> None:
    compose_exec(service, "pkill", "-f", pattern, check=False)


def plan_session(rng: random.Random, hold_out_mutator: str | None, tamper_idx: int | None = None) -> dict:
    """One randomized session plan. `hold_out_mutator`, if set, is excluded (reserved for test).

    `tamper_idx`, if given, cycles tamper segment and mutator evenly (segment varies fastest) so
    the tampered segment is balanced across sessions: with small N a purely random choice skews
    heavily and lets a segment-identity-only model beat chance (go/no-go control)."""
    available = [m for m in MUTATORS if m != hold_out_mutator] or list(MUTATORS)
    is_tamper = rng.random() < 0.6
    if tamper_idx is None:
        segment, mutator = rng.choice(SEGMENT_ORDER), rng.choice(available)
    else:
        segment = SEGMENT_ORDER[tamper_idx % len(SEGMENT_ORDER)]
        mutator = available[(tamper_idx // len(SEGMENT_ORDER)) % len(available)]
    return {
        "tamper_segment": segment if is_tamper else None,
        "mutator": mutator if is_tamper else None,
        "recon": is_tamper and rng.random() < 0.5,
        "n_messages": rng.randint(8, 25),
        "mean_gap": rng.uniform(0.15, 1.2),
        "noise_duration": rng.uniform(10, 25),
        "noise_seed": rng.randint(0, 2**31),
        "sender_seed": rng.randint(0, 2**31),
    }


def start_attacker_relays(run_id: str, plan: dict, ground_truth_dir: str) -> None:
    for segment in SEGMENT_ORDER:
        is_target = segment == plan["tamper_segment"]
        args = [
            "python", "-m", "lab.attacker.proxy",
            "--listen", SEGMENT_IN_ADDR[segment],
            "--downstream", DOWNSTREAM_ADDR[segment],
            "--run-id", run_id,
            "--segment", segment,
            "--ground-truth-path", f"{ground_truth_dir}/{run_id}.jsonl",
        ]
        if is_target:
            args += ["--mutator", plan["mutator"]]
        compose_exec_bg("attacker", *args)
    time.sleep(1.0)  # let all three relays bind before traffic starts


def run_session(run_id: str, plan: dict, pcap_dir: str, ground_truth_dir: str) -> None:
    # "STEP:" lines are parsed by the dashboard's live demo (securetrace.fusion.live) to show progress.
    print(f"STEP: Starting the on-path attacker on segments A, B, C (tampering on "
          f"{plan['tamper_segment'] or 'no segment'}, mutator {plan['mutator'] or 'none'})", flush=True)
    start_attacker_relays(run_id, plan, ground_truth_dir)
    if plan["recon"]:
        print("STEP: Attacker recon: benign-shaped probing of the other segments", flush=True)
        targets = [a for seg, a in SEGMENT_IN_ADDR.items() if seg != plan["tamper_segment"]]
        compose_exec("attacker", "python", "-m", "lab.attacker.recon", "--targets", *targets,
                     "--duration", "3", check=False)

    print("STEP: Starting packet capture on every segment", flush=True)
    compose_exec("sensor", "python3", "lab/sensors/capture.py", "start", "--run-id", run_id,
                 "--pcap-dir", pcap_dir)
    time.sleep(0.5)
    print("STEP: Starting benign background traffic on every segment", flush=True)
    # Benign noise on every segment, both sides of the attacker: in-side clients (sender/relay1/
    # relay2 -> the attacker's address on their segment) and out-side clients (attacker -> the
    # real downstream node), so each segment's capture holds noise flows next to transfer flows.
    duration = str(plan["noise_duration"])
    in_clients = {"A": "sender", "B": "relay1", "C": "relay2"}
    for i, segment in enumerate(SEGMENT_ORDER):
        compose_exec_bg(in_clients[segment], "python", "-m", "lab.agents.noise",
                        "--target", SEGMENT_IN_ADDR[segment], "--duration", duration,
                        "--seed", str(plan["noise_seed"] + i))
        compose_exec_bg("attacker", "python", "-m", "lab.agents.noise",
                        "--target", DOWNSTREAM_ADDR[segment], "--duration", duration,
                        "--seed", str(plan["noise_seed"] + 10 + i))

    print(f"STEP: Sender sends {plan['n_messages']} hop-chained messages; every node reports signed integrity "
          f"events to the collector", flush=True)
    compose_exec("sender", "python", "-m", "lab.agents.sender", "--run-id", run_id,
                 "--n", str(plan["n_messages"]), "--seed", str(plan["sender_seed"]),
                 "--mean-gap", str(plan["mean_gap"]))

    print("STEP: Traffic finished; stopping capture", flush=True)
    time.sleep(1.0)  # let in-flight traffic finish before capture stops
    compose_exec("sensor", "python3", "lab/sensors/capture.py", "stop", "--run-id", run_id,
                 "--pcap-dir", pcap_dir)
    kill_background("attacker", "lab.attacker.proxy")
    for service in ("sender", "relay1", "relay2", "attacker"):
        kill_background(service, "lab.agents.noise")


def extract_features(run_id: str, pcap_dir: str, out_dir: str) -> None:
    print("STEP: Extracting flow features with CICFlowMeter", flush=True)
    compose_exec("sensor", "python3", "lab/sensors/extract.py", "--pcap-dir", pcap_dir,
                 "--out-dir", out_dir, "--run-id", run_id)


def fetch_localizations(run_id: str, retries: int = 10) -> dict:
    """Integrity-derived weak-label source for this run (the collector may lag slightly behind
    the sender finishing, since events are posted asynchronously -- see lab/agents/common.py)."""
    for attempt in range(retries):
        try:
            response = requests.get(f"{COLLECTOR_URL}/runs/{run_id}/localizations", timeout=5)
            if response.ok and response.json():
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return {}


def read_ground_truth(ground_truth_dir: Path, run_id: str) -> list[dict]:
    path = ground_truth_dir / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_log_integrity() -> dict:
    response = requests.get(f"{COLLECTOR_URL}/log/verify", timeout=5)
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "lab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold-out-mutator", choices=sorted(MUTATORS), default=None,
                        help="Exclude this mutator from generation (reserve it for the held-out attack-type test)")
    parser.add_argument("--tamper-offset", type=int, default=0,
                        help="Start of the segment/mutator cycle; use when appending to existing sessions")
    parser.add_argument("--force-tamper", choices=list(SEGMENT_ORDER), default=None,
                        help="Make every session tamper this segment (end-to-end verification)")
    parser.add_argument("--force-clean", action="store_true", help="Make every session a clean (no tamper) one")
    parser.add_argument("--force-mutator", choices=sorted(MUTATORS), default="rewrite_size")
    args = parser.parse_args()

    pcap_dir = args.out_dir / "pcap"
    extracted_dir = pcap_dir / "extracted"
    ground_truth_dir = args.out_dir / "attacker"
    manifest_dir = args.out_dir / "sessions"
    for directory in (pcap_dir, extracted_dir, ground_truth_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    disable_offloads()
    start_benign_servers()
    rng = random.Random(args.seed)
    manifests = []
    tamper_count = 0
    for i in range(args.runs):
        run_id = f"gen{args.seed}-{i:04d}-{uuid.uuid4().hex[:6]}"
        plan = plan_session(rng, args.hold_out_mutator, tamper_idx=tamper_count + args.tamper_offset)
        if args.force_tamper:
            plan.update(tamper_segment=args.force_tamper, mutator=args.force_mutator)
        elif args.force_clean:
            plan.update(tamper_segment=None, mutator=None, recon=False)
        if plan["tamper_segment"] is not None:
            tamper_count += 1
        print(f"[{i + 1}/{args.runs}] run={run_id} tamper={plan['tamper_segment']} mutator={plan['mutator']} "
              f"recon={plan['recon']} n={plan['n_messages']}")
        try:
            run_session(run_id, plan, "/pcap", ground_truth_dir="/out")
            extract_features(run_id, "/pcap", "/pcap/extracted")
        except subprocess.CalledProcessError as exc:
            print(f"  FAILED: {exc.stderr or exc}", file=sys.stderr)
            kill_background("attacker", "lab.attacker.proxy")
            for service in ("sender", "relay1", "relay2", "attacker"):
                kill_background(service, "lab.agents.noise")
            continue

        print("STEP: Collector verifies the hash-chained log and localizes each tamper", flush=True)
        localizations = fetch_localizations(run_id)
        ground_truth = read_ground_truth(ground_truth_dir, run_id)
        manifest = {
            "run_id": run_id,
            "plan": plan,
            "n_messages_acked": sum(1 for entry in ground_truth),
            "localizations": localizations,
            "ground_truth": ground_truth,
            "log_integrity": check_log_integrity(),
        }
        (manifest_dir / f"{run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)

        print(f"STEP: Session saved ({run_id})", flush=True)
        tampered_msgs = sum(1 for e in ground_truth if e["tampered"])
        localized_tamper = sum(1 for v in localizations.values() if v.get("status") == "tamper")
        print(f"  ground truth: {tampered_msgs} tampered / {len(ground_truth)} msgs; "
              f"collector localized tamper on {localized_tamper} msgs; "
              f"log ok={manifest['log_integrity'].get('ok')}")

    index_path = args.out_dir / "runs_index.yaml"
    existing = yaml.safe_load(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    existing += [{"run_id": m["run_id"], "tamper_segment": m["plan"]["tamper_segment"],
                  "mutator": m["plan"]["mutator"]} for m in manifests]
    index_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")

    print(f"\n{len(manifests)}/{args.runs} sessions completed. Index: {index_path}")
    return 0 if len(manifests) == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
