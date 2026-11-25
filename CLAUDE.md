# SecureTrace

Research project: cryptographic integrity checks (hop-chained HMAC) on a multi-hop transfer path produce weak labels and a tampered-segment location, which are used to adapt and focus an ML network-intrusion model. ML is the core of the project; the crypto and Docker lab exist to produce data and labels.

- Idea, novelty claims, datasets, references: [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md)
- Phases, design decisions, verification: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

## Status

**All phases (0-7) are complete.** Final write-up: [FINAL_REPORT.md](FINAL_REPORT.md); verification: `python scripts/verify_project.py` (14/14) and `python scripts/verify_end_to_end.py`; rerun everything with `python scripts/run_all.py`; dashboard: `python scripts/build_dashboard_bundle.py && streamlit run dashboard/app.py`. The older paragraphs below are the build history; where they differ from "Final lab dataset and numbers", the latter is current.

Phase 1 data pipeline and baselines are in place: loader, day-wise splits, five model families (`securetrace/models/factory.py`), metrics, and `scripts/train.py` logging to a local MLflow SQLite store. Primary Q1 config is `configs/phase1_public_baseline.yaml`: train on corrected CIC-IDS2017, test within-dataset (Fri) and cross-dataset on InSDN, all from one Zenodo record so one corrected CICFlowMeter is used throughout. No corrected CSE-CIC-IDS2018 CSV is published, so the 2018 files (original CICFlowMeter) are kept only as the legacy sensitivity config `phase1_public_baseline_ids2018_legacy.yaml`, not for reported Q1 numbers.

Phase 2 (minimal lab + integrity layer) has a working first cut, verified live against the real Docker lab: hop-chained HMAC, signed events, hash-chained log, and segment localization in `securetrace/integrity/`; a FastAPI collector in `lab/collector/`; sender/relay/receiver agents and benign-traffic noise in `lab/agents/`; an on-path TCP-terminating relay attacker (`lab/attacker/proxy.py`, with `flip_byte`/`rewrite_size` mutators and an optional benign-shaped `recon.py` step) wired in by splitting each logical segment into two point-to-point Docker networks so the attacker is a genuine on-path node, not an ARP/off-path trick.

The sensor image (`lab/sensors/`) is built and verified end to end: `capture.py` finds each segment's bridge interfaces by their fixed compose subnet (Docker's auto-generated `br-<hash>` names carry no segment info) and merges each segment's two capture points into one pcap; `extract.py` then runs CICFlowMeter (vendored at `lab/sensors/vendor/cicflowmeter`, a corrected fork — see its LICENSE.txt for the required citation; the one true ML feature extractor, per "One flow extractor" below) and Zeek (`zeek_conn.zeek`, dashboard only) on each pcap. Column names came out already aligned with the Phase 1 public-data schema, with no mapping needed. Getting CICFlowMeter running took several native-library fixes baked into `lab/sensors/Dockerfile`: the vendored `gradlew` has CRLF line endings (stripped at build time, kept LF going forward via `.gitattributes`); the extracted `.so` files lost their executable bit; `libjnetpcap.so`/`libjnetpcap-pcap100.so` need the unversioned `libpcap.so` symlink (Debian's `libpcap0.8` only ships the versioned name) placed specifically in CICFlowMeter's own `lib/native/`, which the native `dlopen("pcap")` call needs; and most notably `cfm`'s generated start script passes the JVM `-Djava.library.path=../lib/native` as a path relative to the *process's cwd*, not to the script's location, so `extract.py` must run it with `cwd=/opt/cicflowmeter/bin`.

`scripts/generate_lab_runs.py` drives real randomized sessions end to end through `docker compose exec` (no reimplemented logic): for each session it picks a tamper segment (or none), a mutator, and whether a recon step precedes it; runs the attacker relay on every segment (pass-through on the two not targeted, since the attacker must sit on the wire for a segment to carry any traffic at all — see `lab/attacker/proxy.py`); captures, extracts (CICFlowMeter + Zeek), and writes one JSON manifest per session joining the collector's integrity-derived localization (the weak-label source) with the attacker's own ground-truth JSONL (`lab/attacker/proxy.py`'s new `GroundTruthLog`, written per message per segment — the TRUE label, used only for evaluation, never training) and the hash-chained log's verification result. `--hold-out-mutator` excludes one mutator from generation, for the held-out-attack-type split. 25/25 generated sessions (so far) have correct localization (every tampered message attributed to its true segment, every clean message `ok`) and a healthy log chain.

**Phase 2 go/no-go check result: GO (second attempt).** The first attempt (25 sessions, original mutators) was NO-GO: held-out PR-AUC 0.213, at chance. The attacker was then given a larger footprint (`rewrite_size` truncates to 10-40% or pads 1-4x; the tampered segment is re-segmented in 512-byte chunks with 0.01 s pacing while pass-through segments stay near-instant; see `lab/attacker/proxy.py`). On 113 sessions, 5 seeds, session-level splits: **held-out PR-AUC 0.934** (GO threshold 0.8). The segment-identity-only control scored 0.163 against ~0.172 chance, so the signal is flow behaviour, not which bridge a flow was captured on. An earlier run of this control scored 0.327 because random tamper-segment choice skewed to C (10 of 17 tampered sessions); `plan_session` now cycles segment and mutator evenly (`--tamper-offset` continues the cycle when appending). Localization vs attacker ground truth: 490/496 messages exact, 0 misattributed; the other 6 are tampered messages the collector marked `incomplete` (a small recall loss for weak labels). NIC-offload disabling is now automated at the start of `generate_lab_runs.py`. Old NO-GO sessions are archived in `data/raw/lab_v1` (gitignored). Note: generation takes about 1.75 min per session, not the ~25 s first assumed.

Caveat: the footprint was made large enough to be learnable; flip_byte tampering alone adds little flow-level trace. Lab traffic is synthetic; do not claim results carry over to a real network.

**Final lab dataset and numbers (100 sessions: 52 tampered, 48 clean; re-run with `python scripts/run_all.py`).** The noise fix: the first 113-session dataset had no benign background flows (noise clients targeted addresses the sender cannot route to), so every flow was a transfer flow and every bag trivially pure; it is archived at `data/raw/lab_v2_nonoise`. Noise listeners now run on the attacker node and noise clients on sender/relay1/relay2 toward the attacker's in-side address and from the attacker toward the relays, so every captured segment holds benign flows (ports 8080/2222/5353) beside transfer flows (9100). Generation takes about 2 min/session. Splits by session: 44 train / 15 val / 41 test, one mutator held out (both `flip_byte` and `rewrite_size` are reported, neither privileged); about 6,100 test flows, attack rate 0.175. All numbers below are 5 seeds, mean (+/- 95% t-interval where given), PR-AUC unless stated, and supersede an earlier 40-session run (where several conclusions were noisier; changes are flagged).

- **Phase 2 re-check, still GO:** true-label model held-out PR-AUC 0.929; segment-identity-only control 0.105 vs ~0.121 chance. True labels need a transfer-port flow near a tampered message (`securetrace/data/lab.py`), so concurrent noise stays benign.
- **Phase 3, weak-label quality** (`scripts/measure_label_quality.py`): bag precision 0.87 / 0.84 / 0.80 / 0.73 / 0.68 and recall 0.98-0.997 at 0.5 / 1 / 2 / 5 / 10 s windows. "Before" (public models on the lab test split, `scripts/evaluate_lab.py`): MLP 0.335 / 0.332, LightGBM 0.27 / 0.25, XGBoost 0.217, logreg 0.21, random forest 0.17 (flip_byte / rewrite_size held out), i.e. near chance for an attack rate of 0.175.
- **Phase 4, weak-label adaptation** (`securetrace/adapt/`, `scripts/adapt_lab.py`, `configs/phase4_adaptation.yaml`): naive XGBoost 0.895 / 0.795 (gain +0.68 +/- 0.11 / +0.58 +/- 0.09); Elkan-Noto XGBoost 0.879 / 0.796; naive MLP 0.819 / 0.851; attention-MIL (bag-level nnPU) 0.686 / 0.638 (gain +0.35 / +0.31); nnPU 0.404 / 0.393 (gain only +0.07); nnPU with labelled-fraction prior 0.596 / 0.562. **Shuffled weak labels remove the gain in all 14 arms and all 5 seeds each.** The naive baseline (unlabelled = negative) was equal or better than every PU/MIL method, in this lab where labels are nearly complete (recall about 0.99); with 100 sessions MIL improved clearly (0.60 -> 0.69) while nnPU got worse (0.61 -> 0.40) and degrades as more tampered sessions are added (0.60 at 25% of them, 0.40 at 100%), consistent with the over-estimated class prior (Elkan-Noto c about 0.62-0.65 vs label recall about 0.98; the estimator is biased low when positives overlap other flows, see `tests/test_adapt.py`). XGBoost saturates by 50% of the tampered sessions (about 26): 0.73 / 0.63 at 25%, 0.89 / 0.83 at 50%. Window width: XGBoost is robust (0.86-0.88 across 0.5-10 s for flip_byte, 0.88 -> 0.76 for rewrite_size); MIL/nnPU degrade with wide windows as bag precision falls.
- **Phase 5, label-free comparators** (`securetrace/unsup/`, `scripts/compare_unsup_lab.py`): isolation forest 0.21 / 0.25, autoencoder 0.23 / 0.39, CORAL 0.20 / 0.21, SSL encoder + public head 0.13 / 0.15 (below chance), DANN 0.45 / 0.31 (best label-free but very unstable, 95% CI about +/-0.25). Weak-label adaptation (0.8-0.9) beats all of them. SCARF pretraining adds nothing to weak-label fine-tuning: paired difference +0.003 +/- 0.018 and -0.017 +/- 0.014; the from-scratch network equals the public-initialised one, so public pretraining is not what drives the gain. The optional sequence-model stretch was not done; comparators were not hyper-parameter tuned.
- **Phase 6, localization and fusion** (`securetrace/fusion/`, `scripts/fuse_lab.py`): localization 52/52 tampered sessions correct, 0 misattributions, 0 false tamper calls on clean sessions. On the 31 test incidents per split, irrelevant alerts per incident (alert = calibrated p >= 0.5): ML-only 1.27 / 0.99, time-window correlation (any segment) 1.23 / 0.94, **localized fusion 0.10 / 0.10**, at the same recall (0.68 / 0.66); false alerts per clean session 0 for localized and time-window vs 0.14 / 0.10 for ML-only. precision@5/@10 are about 1.0 in every mode (ceiling), so they are not evidence for Novelty B. Calibration (on weak validation labels), ECE vs true labels: isotonic 0.017 / 0.033, raw 0.028 / 0.049, temperature 0.020 / 0.054: a modest isotonic improvement, within the intervals' overlap; in the earlier 40-session run calibration had made ECE worse, so treat the calibration benefit as unproven. TreeSHAP from XGBoost `pred_contribs` (the `shap` package is not installed); example alert in `data/processed/phase6_example_alert.json`. Note the ML-only irrelevant-alert count was 10 per incident for `rewrite_size` in the 40-session run and is 1 here: that earlier figure was a small-sample artefact.
- **Caveats for all of the above:** lab traffic is synthetic; one attacker implementation, one noise generator, and train/test share the lab, so nothing here shows transfer to a real network; the `rewrite_size` footprint was deliberately enlarged; "irrelevant" is judged against the attacker's own flow labels; Novelty A and B are variants/combinations of prior work, not "first".

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
