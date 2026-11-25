# SecureTrace

Research project: cryptographic integrity checks (hop-chained HMAC) on a multi-hop transfer path produce weak labels and a tampered-segment location, which are used to adapt and focus an ML network-intrusion model. ML is the core of the project; the crypto and Docker lab exist to produce data and labels.

- Idea, novelty claims, datasets, references: [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md)
- Phases, design decisions, verification: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

## Status

Phase 1 data pipeline and baselines are in place: loader, day-wise splits, five model families (`securetrace/models/factory.py`), metrics, and `scripts/train.py` logging to a local MLflow SQLite store. Primary Q1 config is `configs/phase1_public_baseline.yaml`: train on corrected CIC-IDS2017, test within-dataset (Fri) and cross-dataset on InSDN, all from one Zenodo record so one corrected CICFlowMeter is used throughout. No corrected CSE-CIC-IDS2018 CSV is published, so the 2018 files (original CICFlowMeter) are kept only as the legacy sensitivity config `phase1_public_baseline_ids2018_legacy.yaml`, not for reported Q1 numbers.

Phase 2 (minimal lab + integrity layer) has a working first cut, verified live against the real Docker lab: hop-chained HMAC, signed events, hash-chained log, and segment localization in `securetrace/integrity/`; a FastAPI collector in `lab/collector/`; sender/relay/receiver agents and benign-traffic noise in `lab/agents/`; an on-path TCP-terminating relay attacker (`lab/attacker/proxy.py`, with `flip_byte`/`rewrite_size` mutators and an optional benign-shaped `recon.py` step) wired in by splitting each logical segment into two point-to-point Docker networks so the attacker is a genuine on-path node, not an ARP/off-path trick.

The sensor image (`lab/sensors/`) is built and verified end to end: `capture.py` finds each segment's bridge interfaces by their fixed compose subnet (Docker's auto-generated `br-<hash>` names carry no segment info) and merges each segment's two capture points into one pcap; `extract.py` then runs CICFlowMeter (vendored at `lab/sensors/vendor/cicflowmeter`, a corrected fork — see its LICENSE.txt for the required citation; the one true ML feature extractor, per "One flow extractor" below) and Zeek (`zeek_conn.zeek`, dashboard only) on each pcap. Column names came out already aligned with the Phase 1 public-data schema, with no mapping needed. Getting CICFlowMeter running took several native-library fixes baked into `lab/sensors/Dockerfile`: the vendored `gradlew` has CRLF line endings (stripped at build time, kept LF going forward via `.gitattributes`); the extracted `.so` files lost their executable bit; `libjnetpcap.so`/`libjnetpcap-pcap100.so` need the unversioned `libpcap.so` symlink (Debian's `libpcap0.8` only ships the versioned name) placed specifically in CICFlowMeter's own `lib/native/`, which the native `dlopen("pcap")` call needs; and most notably `cfm`'s generated start script passes the JVM `-Djava.library.path=../lib/native` as a path relative to the *process's cwd*, not to the script's location, so `extract.py` must run it with `cwd=/opt/cicflowmeter/bin`.

`scripts/generate_lab_runs.py` drives real randomized sessions end to end through `docker compose exec` (no reimplemented logic): for each session it picks a tamper segment (or none), a mutator, and whether a recon step precedes it; runs the attacker relay on every segment (pass-through on the two not targeted, since the attacker must sit on the wire for a segment to carry any traffic at all — see `lab/attacker/proxy.py`); captures, extracts (CICFlowMeter + Zeek), and writes one JSON manifest per session joining the collector's integrity-derived localization (the weak-label source) with the attacker's own ground-truth JSONL (`lab/attacker/proxy.py`'s new `GroundTruthLog`, written per message per segment — the TRUE label, used only for evaluation, never training) and the hash-chained log's verification result. `--hold-out-mutator` excludes one mutator from generation, for the held-out-attack-type split. 25/25 generated sessions (so far) have correct localization (every tampered message attributed to its true segment, every clean message `ok`) and a healthy log chain.

**Phase 2 go/no-go check result: GO (second attempt).** The first attempt (25 sessions, original mutators) was NO-GO: held-out PR-AUC 0.213, at chance. The attacker was then given a larger footprint (`rewrite_size` truncates to 10-40% or pads 1-4x; the tampered segment is re-segmented in 512-byte chunks with 0.01 s pacing while pass-through segments stay near-instant; see `lab/attacker/proxy.py`). On 113 sessions, 5 seeds, session-level splits: **held-out PR-AUC 0.934** (GO threshold 0.8). The segment-identity-only control scored 0.163 against ~0.172 chance, so the signal is flow behaviour, not which bridge a flow was captured on. An earlier run of this control scored 0.327 because random tamper-segment choice skewed to C (10 of 17 tampered sessions); `plan_session` now cycles segment and mutator evenly (`--tamper-offset` continues the cycle when appending). Localization vs attacker ground truth: 490/496 messages exact, 0 misattributed; the other 6 are tampered messages the collector marked `incomplete` (a small recall loss for weak labels). NIC-offload disabling is now automated at the start of `generate_lab_runs.py`. Old NO-GO sessions are archived in `data/raw/lab_v1` (gitignored). Note: generation takes about 1.75 min per session, not the ~25 s first assumed.

Caveat: the footprint was made large enough to be learnable; flip_byte tampering alone adds little flow-level trace. Lab traffic is synthetic; do not claim results carry over to a real network.

**Phase 2 re-check on noisy data: still GO.** The 113-session dataset above contained no benign background flows at all (noise clients targeted addresses the sender cannot route to, so every lab flow was a transfer flow and every weak-label bag was trivially pure). Fixed: noise listeners run on the attacker node, noise clients run on sender/relay1/relay2 toward the attacker's in-side address and from the attacker toward the relays, so each captured segment holds benign flows (ports 8080, 2222, 5353) beside transfer flows (port 9100). That dataset is archived at `data/raw/lab_v2_nonoise` (gitignored). On the 40-session noisy dataset (24 tampered, 16 clean; generation stopped early to save time, ~2 min/session): held-out PR-AUC **0.922**, segment-only control **0.132** vs ~0.141 chance.

**Phase 3 done (lab dataset, weak-label builder, label quality, "before" numbers).** True labels now require a transfer-port flow near a tampered message (`securetrace/data/lab.py`), so concurrent noise stays benign; weak labels come only from the collector's `localizations` (`securetrace/data/weak_labels.py`, key/config errors and `incomplete` excluded). Session-level split with one mutator held out (`securetrace/data/lab_split.py`). Label quality vs true labels by window width (`scripts/measure_label_quality.py`): bag precision 0.88 / 0.85 / 0.80 / 0.73 / 0.69 and recall 0.98-1.00 at 0.5 / 1 / 2 / 5 / 10 s. "Before" numbers (`scripts/evaluate_lab.py`, public-data models scored on the lab test split, 5 seeds, `data/processed/phase3_before_numbers.csv`): mean PR-AUC 0.17-0.35 against an attack rate of 0.18-0.19, i.e. near chance for every family (best: MLP 0.35/0.35 for held-out rewrite_size/flip_byte; random forest worst at 0.17-0.18). This is the cross-network gap Phase 4 tries to close. Only 40 sessions exist so far, so label-efficiency curves will be coarse; generate more with `generate_lab_runs.py --tamper-offset <n>` when time allows. `train.py --save-model` now serializes with cloudpickle (skops rejected our own pipeline classes).

**Phase 4 done (Novelty A: weak-label adaptation, Q2).** `securetrace/adapt/` (data, PU/Elkan-Noto/nnPU, XGBoost continuation boosting, MLP-from-public-weights with naive / nnPU / attention-MIL with a bag-level nnPU risk), `scripts/adapt_lab.py --config configs/phase4_adaptation.yaml` (5 seeds, both held-out mutators, MLflow experiment `securetrace-phase4`, rows in `data/processed/phase4_results.csv`), `scripts/summarize_phase4.py` (paired per-seed gain over the Phase 3 "before" numbers, 95% t-interval). Training uses only weak labels from integrity failures; thresholds come from weak validation labels; true labels score the lab test split only. Results (PR-AUC, 40 sessions, 18 train / 5 val / 17 test, held-out mutator flip_byte / rewrite_size, before -> after, gain +/- 95% CI over 5 seeds):
- XGBoost: before 0.22 / 0.24; naive 0.89 / 0.81 (+0.67 +/- 0.09 / +0.58 +/- 0.11); Elkan-Noto 0.89 / 0.77 (+0.67 +/- 0.09 / +0.53 +/- 0.19).
- MLP: before 0.35; naive 0.81 / 0.83; nnPU 0.61 / 0.51; attention-MIL 0.60 / 0.62 (gains +0.16 to +0.26 with CIs of +/-0.2 to +/-0.3, i.e. noisy).
- **Shuffled-label control passes**: shuffled weak labels erase the gain in every arm (XGBoost gain about 0.0, 0.19-0.24 PR-AUC vs 0.77-0.89 real; MLP below before). nnPU's gain vanished in 4 of 5 seeds per held-out mutator (still clearly below real in the fifth).
- **Honest reading: the PU and MIL methods did NOT beat the naive baseline here.** Naive (unlabelled treated as negative) was equal or better for both families. Likely reason: the lab labels are nearly complete (bag recall 0.98-1.0, precision 0.69-0.88), so unlabelled flows really are almost all benign and the naive assumption holds in this synthetic lab; PU/MIL machinery pays a variance cost for protection that is not needed. This is a result about this lab, not evidence that PU/MIL are useless in general.
- Class-prior estimation: the Elkan-Noto estimate gave c about 0.62-0.65 (prior about 0.21) while the real malicious fraction is about 0.11 and label recall is about 0.98. A synthetic check shows the estimator is biased low whenever positives overlap other flows, independent of regularization (tests/test_adapt.py pins this), so the prior overshoots. The `nnpu_lf`/`mil_lf` ablation (prior = labelled fraction) was added after seeing this; it helped nnPU somewhat (0.54 / 0.67) but is a post-hoc ablation, not a tuned result. Hyper-parameters (lr 3e-4, 40 epochs) were fixed before looking at test scores and not tuned.
- Window width: XGBoost is robust (0.86-0.90 at 0.5-10 s); the neural methods degrade with wide windows (MIL 0.64 -> 0.36 at 10 s for flip_byte), tracking the falling bag precision.
- Label efficiency: XGBoost saturates already at 25% of the tampered training sessions (about 3 sessions); the neural methods need all of them (gain near 0 at 25-50%). With only 18 training sessions these curves are coarse.
- Caveats: lab traffic is synthetic; test sessions share the lab, attacker implementation and noise generator with training, so these numbers do not show transfer to a real network. The `rewrite_size` footprint was deliberately enlarged. Claim Novelty A only as a variant/combination of prior PU/MIL work.

**Phase 5 done (label-free comparators, Q3).** `securetrace/unsup/` (isolation forest + autoencoder on unlabelled lab train flows; SCARF-style contrastive pretraining; CORAL; DANN), `scripts/compare_unsup_lab.py --config configs/phase5_label_free.yaml` (same lab split, 5 seeds, both held-out mutators, MLflow `securetrace-phase5`, `data/processed/phase5_results.csv`), `scripts/summarize_phase5.py`. Domain-adaptation source = 40k labelled public training-day flows; columns the lab never populates are dropped for these methods. PR-AUC, mean over 5 seeds (flip_byte / rewrite_size held out; attack rate about 0.18-0.19; public model before: MLP 0.35 / 0.35, XGBoost 0.22 / 0.24):
- Label-free: isolation forest 0.21 / 0.27, autoencoder 0.22 / 0.33, CORAL 0.19 / 0.20, SSL encoder + public-data head 0.14 / 0.18 (below chance), DANN 0.31 / 0.45 (best label-free but very unstable, 95% CI about +/-0.25). None reliably beats the unadapted public model.
- Weak-label adaptation (Phase 4): 0.8-0.9 for naive XGBoost/MLP. So integrity-derived weak labels beat every label-free comparator here by a wide margin.
- "Best combination" (SCARF-pretrained encoder + weak-label fine-tune) 0.80 / 0.83 vs the same network from random init 0.81 / 0.84: SSL pretraining contributes nothing measurable (paired difference -0.010 +/- 0.022 and -0.013 +/- 0.031). The scratch network also matches Phase 4's public-initialised fine-tune, so with this much weak-labelled lab data the public pretraining is not what drives the gain.
- Caveats: anomaly detectors were fit on lab flows contaminated with attacks (nothing says which are clean); only 18 training sessions; the optional GRU/transformer sequence-model stretch was NOT done. Comparators were not hyper-parameter tuned, so "label-free methods fail" means these off-the-shelf configurations fail on this small synthetic lab, not that the methods cannot work.

**Phase 6 done (Novelty B: localization and fusion, Q4).** `securetrace/fusion/` (`calibrate.py`: isotonic + temperature scaling, ECE, reliability bins; `alerts.py`: three alerting modes, precision@k, alert/irrelevant-alert counts, XGBoost TreeSHAP via `pred_contribs`), `scripts/fuse_lab.py --config configs/phase6_fusion.yaml`, `scripts/summarize_phase6.py`; outputs in `data/processed/phase6_*` (results, calibration, localization, reliability PNGs, `phase6_example_alert.json`). The scoring model is the Phase 4 weak-label-adapted XGBoost; calibrators are fit on WEAK validation labels; an "incident" is a test session where the collector localized a tamper (14 per held-out split). Results (5 seeds, flip_byte / rewrite_size held out):
- **Localization:** 24/24 tampered sessions attributed to the true segment, 0 misattributions, 0 false tamper calls on clean sessions.
- **Irrelevant alerts per incident** (alert = calibrated p >= 0.5): ML-only 1.00 / 10.1, time-window correlation (flows on any segment overlapping an integrity-alert window) 0.94 / 9.07, **localized fusion 0.16 / 0.33**; alert recall is the same for all three (0.69 / 0.73), so localization removes the irrelevant alerts without losing true ones. False alerts per CLEAN session: localized and time-window 0 (no integrity failure, no alert); ML-only 0.40 / 2.07.
- **precision@5 and @10 are 1.0 in every mode**: inside an incident session the adapted model already ranks real attack flows first, so these metrics hit a ceiling and do not discriminate; the benefit shows only in alert counts. Do not report precision@k as evidence for Novelty B.
- **Calibration did NOT help.** ECE vs true labels: raw 0.019 / 0.075, isotonic 0.034 / 0.102, temperature 0.038 / 0.096. Calibrating on weak validation labels (bag precision 0.8, so the labels overstate the malicious rate) moved probabilities away from the truth, and the calibrated threshold produced more irrelevant alerts than the raw one (0.16 vs 0.03 per incident for flip_byte). The shuffled-label and calibration findings point the same way: the weak labels are good for ranking, not for absolute probabilities. A first version of the time-window baseline accidentally reused the localized segment's flows and so matched `localized` exactly; it was fixed to take flows on any segment and re-run.
- SHAP: exact TreeSHAP values from XGBoost (`shap` package not installed); example fused alert with integrity evidence and top features in `phase6_example_alert.json`.
- Caveats: 14 incidents per split, a single synthetic lab, one attacker implementation; "irrelevant" is judged against the attacker's own flow labels. Claim Novelty B only as a combination of prior localization and alert-correlation ideas, and do not state that results carry over to a real network.

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
