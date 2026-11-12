# SecureTrace: ML-heavy development plan

## Context

[SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md) describes the idea and treats ML as one component (a single XGBoost model plus a PU fine-tune in week 3). This plan makes ML the core of the project.

**The cryptography and the lab become the data and label source, and the ML becomes the research.** The integrity layer is kept deliberately small (about 15% of the effort). The main result is a comparison study answering one question:

> Can cryptographically verified integrity failures replace manual labels for adapting an intrusion-detection model to a new network, and how do they compare with other ways of adapting?

Estimated duration: about 8 weeks at part-time pace (the original plan was 4). Phases 5 to 7 are separable, so the scope can be cut without breaking what comes before.

## The four ML questions

| # | Question | Result it produces |
|---|---|---|
| Q1 | How badly does a public-dataset model fail on another network? | Cross-dataset gap table (2017 → 2018 → lab) |
| Q2 | Do integrity-derived weak labels close that gap? (Novelty A) | Before/after table, PU vs MIL vs naive fine-tuning |
| Q3 | How do they compare with label-free adaptation? | Same table with self-supervised, anomaly-detection and domain-adaptation rows |
| Q4 | Does localizing to a segment and fusing with the ML score give better alerts? (Novelty B) | Ranking metrics per incident, with and without localization |

## Three design decisions that shape everything

1. **One flow extractor everywhere: CICFlowMeter (the corrected version).** The corrected CIC-IDS2017/2018 CSVs were produced with it, so the lab sensors capture pcaps and run the same tool. This avoids re-extracting the public pcaps. Zeek still runs in the lab, but only for the dashboard, not as model input.
2. **Weak labels are bag-level, not flow-level.** An HMAC failure says "something on segment B between t1 and t2 was malicious", not which flow. That is a multiple-instance learning (MIL) problem as much as a PU problem, so both are implemented and compared.
3. **The lab records true per-flow labels.** The attacker container logs its own 5-tuples and timestamps. These are never used for training. They are used only to score the weak-label methods, which is what makes Q2 measurable.

## Biggest risk (to settle in phase 2, not week 6)

A proxy that only flips payload bytes may leave **no trace in flow metadata**. Then the model either learns nothing or learns lab artefacts such as "segment B is always the bad one". Mitigations built into the plan:

- The attacker has several realistic implementations with a network footprint: ARP-spoof MITM, TCP-terminating proxy (adds latency and re-segmentation), size-changing rewrite, each with optional preceding recon.
- Tampered segment, timing and benign traffic mix are randomized per run.
- Identity features (IPs, ports of lab hosts, segment id) are dropped from model input.
- One attack type is held out entirely from training and used only for testing.
- Early go/no-go check: a model trained on the *true* lab labels must separate attack from benign flows. If it cannot, the attacks need more footprint before any weak-label work starts.

## Repo layout

```text
SecureTrace/
  lab/            docker-compose.yml, agents/, attacker/, sensors/, collector/
  securetrace/
    data/         loaders, cleaning, splits, weak-label builder
    features/     shared feature schema + preprocessing
    models/       baselines, deep tabular, PU, MIL, DA, anomaly
    fusion/       calibration, alert ranking
    eval/         metrics, control checks
  configs/        one YAML per experiment
  scripts/        train.py, evaluate.py, generate_lab_runs.py
  dashboard/      Streamlit app
  tests/
  data/           (gitignored) raw/, interim/, processed/
```

Stack: Python 3.11, PyTorch (MPS on the Mac), scikit-learn, XGBoost/LightGBM, MLflow for experiment tracking, FastAPI, SQLite, Docker Compose, Streamlit, pytest.

## Phases

### Phase 0: Setup (2 days)
- `git init` inside `SecureTrace/`. Git currently treats the whole home directory as the repo, so this project needs its own.
- Python environment, layout above, `.gitignore` for data and keys, MLflow running locally.
- Fixed seeds and one config file per experiment from the first run on.

### Phase 1: Public data and baselines, Q1 (week 1 to 2)
- Download corrected CIC-IDS2017 and a subset of days from corrected CSE-CIC-IDS2018.
- Cleaning: drop identifiers and timestamps from features, handle inf/NaN, remove duplicates, fix the feature schema in `securetrace/features/`.
- Splits by day, never by random row.
- Models: logistic regression, random forest, XGBoost, LightGBM, a plain MLP.
- Output: within-dataset vs cross-dataset table (macro-F1, PR-AUC, FPR at fixed recall). This is the gap the rest of the project tries to close.

### Phase 2: Minimal lab and integrity layer (week 2 to 3, overlaps phase 1)
- Docker Compose: sender, relay1, relay2, receiver, attacker, per-segment pcap capture, collector.
- Agents: SHA-256 + hop-chained HMAC `tag_i = HMAC(k_i, data || tag_{i-1})`, signed events to a FastAPI collector with a hash-chained SQLite log. Keys from env only.
- Attacker variants and benign traffic generator as described under "Biggest risk".
- `generate_lab_runs.py`: runs N randomized sessions and writes pcaps, CICFlowMeter CSVs, integrity events and true labels.
- Run the go/no-go check.

### Phase 3: Lab dataset and weak-label builder (week 3 to 4)
- Generate a few hundred sessions; split by session into train / validation / test, with one attack type held out.
- Weak-label builder: integrity failure → (segment, time window) → positive bag; everything else unlabelled. Key and config errors are excluded.
- Measure label quality against the true labels: what fraction of flows in a positive bag are actually malicious, as a function of window width.
- Evaluate the phase 1 models on the lab test set: the "before" numbers.

### Phase 4: Novelty A, weak-label adaptation, Q2 (week 4 to 5)
Fine-tune the public-data model with the weak labels, three ways:
- **Naive:** treat unlabelled as negative (the baseline to beat).
- **PU learning:** Elkan-Noto reweighting for XGBoost, nnPU loss for the neural model, with class-prior estimation.
- **MIL:** attention-pooling network over the flows in a bag, trained on bag labels, read out per-flow scores.

Studies: label-efficiency curve (performance vs number of tamper incidents), sensitivity to window width, and the shuffled-label control (the gain must vanish).

### Phase 5: Label-free comparators, Q3 (week 5 to 6)
- **Anomaly detection:** isolation forest and an autoencoder trained on lab traffic.
- **Self-supervised pretraining:** masked-feature or contrastive (SCARF-style) pretraining of a tabular encoder on unlabelled lab flows, then the classifier head from public data.
- **Unsupervised domain adaptation:** CORAL and DANN from public data to the lab.
- Best combination: self-supervised encoder + weak-label fine-tuning.
- Optional stretch: a sequence model (GRU or small transformer) over consecutive flows per segment, to use temporal context.

### Phase 6: Novelty B, localization and fusion, Q4 (week 6 to 7)
- Localize the tampered segment from the HMAC chain.
- Calibrate model scores (isotonic / temperature scaling; report ECE and reliability plots).
- Fused alert: rank that segment's flows by calibrated score, attach the integrity evidence.
- Compare three alerting modes: ML only on all segments, time-window correlation (how SIEMs do it), and localized fusion. Metrics: precision@k, irrelevant alerts per incident, localization accuracy.
- SHAP explanations for the top flows in an alert.

### Phase 7: Dashboard and write-up (week 7 to 8)
- Streamlit with the four panels (Integrity, ML risk, Tampered segment, Fused alert), reading from the collector DB and a saved model.
- Final results tables straight from MLflow runs, each with confidence intervals over at least 5 seeds.
- Limitations section: lab traffic is synthetic; results do not transfer to a real network without validation there.

## Verification

- **Unit tests (pytest):** flipping one payload byte fails HMAC verification; a tamper on segment k is attributed to k for every k; the event log detects a modified row; the weak-label builder excludes key/config errors.
- **Leakage checks:** no identifier or timestamp columns in the feature matrix; no session appears in two splits; the held-out attack type is absent from training.
- **Controls that must be able to fail:** shuffled weak labels remove the Novelty A gain; a model trained on segment id alone scores at chance on the test set; the go/no-go check in phase 2.
- **Reproducibility:** `python scripts/train.py --config configs/<exp>.yaml` reproduces any table row from a clean checkout plus the data.
- **End to end:** `docker compose up`, run one tamper session, and see the integrity failure, the localized segment and the ranked alert appear in the dashboard.

## Next step

Phase 0 and the phase 1 data pipeline: repo scaffold, dataset loaders and cleaning, day-wise splits, and the first XGBoost baseline logged to MLflow.
