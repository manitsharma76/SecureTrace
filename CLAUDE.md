# SecureTrace

Research project: cryptographic integrity checks (hop-chained HMAC) on a multi-hop transfer path produce weak labels and a tampered-segment location, which are used to adapt and focus an ML network-intrusion model. ML is the core of the project; the crypto and Docker lab exist to produce data and labels.

- Idea, novelty claims, datasets, references: [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md)
- Phases, design decisions, verification: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

## Status

Phase 1 data pipeline and baselines are in place: loader, day-wise splits, five model families (`securetrace/models/factory.py`), metrics, and `scripts/train.py` logging to a local MLflow SQLite store. Primary Q1 config is `configs/phase1_public_baseline.yaml`: train on corrected CIC-IDS2017, test within-dataset (Fri) and cross-dataset on InSDN, all from one Zenodo record so one corrected CICFlowMeter is used throughout. No corrected CSE-CIC-IDS2018 CSV is published, so the 2018 files (original CICFlowMeter) are kept only as the legacy sensitivity config `phase1_public_baseline_ids2018_legacy.yaml`, not for reported Q1 numbers.

Phase 2 (minimal lab + integrity layer) has a working first cut, verified live against the real Docker lab: hop-chained HMAC, signed events, hash-chained log, and segment localization in `securetrace/integrity/`; a FastAPI collector in `lab/collector/`; sender/relay/receiver agents and benign-traffic noise in `lab/agents/`; an on-path TCP-terminating relay attacker (`lab/attacker/proxy.py`, with `flip_byte`/`rewrite_size` mutators and an optional benign-shaped `recon.py` step) wired in by splitting each logical segment into two point-to-point Docker networks so the attacker is a genuine on-path node, not an ARP/off-path trick.

The sensor image (`lab/sensors/`) is built and verified end to end: `capture.py` finds each segment's bridge interfaces by their fixed compose subnet (Docker's auto-generated `br-<hash>` names carry no segment info) and merges each segment's two capture points into one pcap; `extract.py` then runs CICFlowMeter (vendored at `lab/sensors/vendor/cicflowmeter`, a corrected fork — see its LICENSE.txt for the required citation; the one true ML feature extractor, per "One flow extractor" below) and Zeek (`zeek_conn.zeek`, dashboard only) on each pcap. Column names came out already aligned with the Phase 1 public-data schema, with no mapping needed. Getting CICFlowMeter running took several native-library fixes baked into `lab/sensors/Dockerfile`: the vendored `gradlew` has CRLF line endings (stripped at build time, kept LF going forward via `.gitattributes`); the extracted `.so` files lost their executable bit; `libjnetpcap.so`/`libjnetpcap-pcap100.so` need the unversioned `libpcap.so` symlink (Debian's `libpcap0.8` only ships the versioned name) placed specifically in CICFlowMeter's own `lib/native/`, which the native `dlopen("pcap")` call needs; and most notably `cfm`'s generated start script passes the JVM `-Djava.library.path=../lib/native` as a path relative to the *process's cwd*, not to the script's location, so `extract.py` must run it with `cwd=/opt/cicflowmeter/bin`.

61 tests pass (`tests/test_integrity.py`, `tests/test_collector.py`, `tests/test_attacker.py`, `tests/test_sensors.py`). Not yet done: `scripts/generate_lab_runs.py` (still a stub, now able to call `lab/sensors/capture.py` + `extract.py` directly), the go/no-go check, and automating the NIC-offload fix (currently applied by hand after `docker compose up`) into lab startup.

## Target layout

```text
lab/            docker-compose.yml, agents/, attacker/, sensors/, collector/
securetrace/    data/, features/, models/, fusion/, eval/
configs/        one YAML per experiment
scripts/        train.py, evaluate.py, generate_lab_runs.py
dashboard/      Streamlit app
tests/          pytest
data/           gitignored: raw/, interim/, processed/
```

Stack: Python 3.11, PyTorch (MPS), scikit-learn, XGBoost/LightGBM, MLflow, FastAPI, SQLite, Docker Compose, Streamlit, pytest.

## Planned commands

```bash
python scripts/train.py --config configs/<exp>.yaml   # one experiment, logged to MLflow
python scripts/generate_lab_runs.py                    # randomized lab sessions
pytest                                                 # unit, leakage and control checks
docker compose -f lab/docker-compose.yml up            # the lab
```

## Rules that must not be broken

ML methodology:
- **One flow extractor.** Model input is always corrected-CICFlowMeter features, for public data and lab data alike. Zeek output is for the dashboard only.
- **Split by day (public data) or by session (lab data).** Never by random row. No session may appear in two splits.
- **True lab labels are for evaluation only.** Training on the lab uses only weak labels derived from integrity failures.
- **No identity features.** Drop IPs, lab host ports, segment id and timestamps from the feature matrix.
- **One attack type stays held out** of training and is used only for testing.
- **Weak labels are bag-level:** (segment, time window) is positive, everything else is unlabelled, not negative. Key and config errors are excluded from labels.
- **Every experiment has a config file and a fixed seed** and is logged to MLflow. Reported numbers come from logged runs, over at least 5 seeds.
- **Keep the controls working:** shuffled weak labels must remove the Novelty A gain; a tamper on segment k must never be attributed to another segment; a one-byte payload flip must fail verification.

Security:
- Keys come from environment variables or a secrets file. Never put them in code, logs, configs or commits.
- Attacker code runs only inside the isolated Docker network. Never point it at a real network.
- Store flow metadata only, no payloads.

Claims:
- Novelty A and B are claimed as variants/combinations of prior work, not as "first". Keep that wording in any write-up.
- Lab traffic is synthetic. Do not state that results carry over to a real network.

## Conventions

- Data files, pcaps, models and MLflow runs are gitignored.
- New model families go in `securetrace/models/` behind the same train/predict interface so `scripts/train.py` can run any of them from a config.
- Add a test alongside any change to the integrity layer, the split logic or the weak-label builder.
