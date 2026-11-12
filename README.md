# SecureTrace

Integrity-anchored network intrusion detection.

SecureTrace checks data integrity with cryptography at every checkpoint on a transfer path, then uses those results to drive a machine-learning intrusion model:

- **Cryptography answers:** did the data change, and on which segment?
- **ML answers:** what suspicious network activity happened on that segment?

> **Status: planning.** No code has been written yet. This README describes what the project will build. See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the schedule.

## Why

- A failed HMAC proves data changed, but says nothing about how or by whom.
- ML intrusion detectors trained on public datasets score very well on that dataset and fall to about chance level on a different network. Labelled data from your own network is rarely available.
- Existing tools correlate host and network alerts by time window. They do not use cryptographic verification as ground truth, and they do not localize tampering to a path segment.

## The idea

An integrity failure is a label with almost no false positives. SecureTrace uses it in two ways:

1. **Integrity-anchored weak supervision.** Flows on the failing segment, inside the failure's time window, are marked suspicious; everything else stays unlabelled. A model trained on public data is then adapted to the local network with these labels, with no manual labelling.
2. **Checkpoint-localized correlation.** A hop-chained HMAC shows which segment the data changed on. Only that segment's flows are scored, and the integrity result and the model score are fused into one ranked alert.

Both are variants and combinations of earlier work, not claimed as firsts. Prior work is listed in [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md).

## Architecture

```text
Sender ──► Relay 1 ──► Relay 2 ──► Receiver
  │   seg A   │   seg B   │   seg C   │
  └─ hop-chained HMAC checked at every checkpoint ─┘
         │                │               │
     sensor A         sensor B        sensor C      (flow metadata per segment)
         │                │               │
         └──────────► Collector (FastAPI, hash-chained event log)
                              │
                 ML models ◄── weak labels from integrity failures
                              │
                     Streamlit dashboard
```

Everything runs in an isolated Docker Compose lab that includes an attacker container for generating tampering runs.

## Research questions

| # | Question |
|---|---|
| Q1 | How badly does a public-dataset model fail on another network? |
| Q2 | Do integrity-derived weak labels close that gap? |
| Q3 | How do they compare with label-free adaptation (anomaly detection, self-supervised pretraining, domain adaptation)? |
| Q4 | Does localizing to a segment and fusing with the ML score give better alerts? |

## Methods

- **Baselines:** logistic regression, random forest, XGBoost, LightGBM, MLP.
- **Weak-label adaptation:** PU learning (Elkan-Noto, nnPU) and multiple-instance learning with attention pooling.
- **Comparators:** isolation forest, autoencoder, self-supervised tabular pretraining, CORAL, DANN.
- **Alerting:** score calibration, localized fusion, SHAP explanations.

## Datasets

| Purpose | Dataset |
|---|---|
| Base training | CIC-IDS2017, corrected version (Engelen et al., 2021) |
| Cross-dataset test | CSE-CIC-IDS2018, corrected version (Liu, Engelen et al., 2022) |
| Adaptation and localization | Own lab dataset, generated in the Docker lab |

All model input uses the same flow extractor (corrected CICFlowMeter) so features are comparable across datasets.

## Roadmap

| Phase | Work |
|---|---|
| 0 | Repo setup, environment, experiment tracking |
| 1 | Public data pipeline and baselines (Q1) |
| 2 | Docker lab and integrity layer |
| 3 | Lab dataset and weak-label builder |
| 4 | Weak-label adaptation (Q2) |
| 5 | Label-free comparators (Q3) |
| 6 | Localization and fusion (Q4) |
| 7 | Dashboard and write-up |

Details, risks and verification checks are in [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).

## Tech stack

Python 3.11, PyTorch, scikit-learn, XGBoost, LightGBM, MLflow, FastAPI, SQLite, Docker Compose, CICFlowMeter, Zeek, Streamlit, pytest.

## Security and ethics

- The lab and its attacker run only on an isolated Docker network, never on real or unauthorized networks.
- Keys are never hard-coded or logged.
- Only flow metadata is stored, no payloads.
- Lab traffic is synthetic. Results do not carry over to a real network without validating on that network.

## Documents

- [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md): idea, novelty, datasets, references
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md): ML-heavy development plan
