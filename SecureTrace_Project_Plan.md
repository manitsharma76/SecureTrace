# SecureTrace: Integrity-Anchored Network Intrusion Detection

## 1. Idea in one line

SecureTrace checks data integrity with cryptography at every checkpoint on a transfer path. It uses those integrity results to (a) find the segment where tampering happened and (b) teach an ML network-intrusion model to work on our own network.

> Cryptography answers: *Did the data change, and on which segment?*
> ML answers: *What suspicious network activity happened on that segment?*

## 2. Problem

- **Crypto alone:** an HMAC or signature failure proves data changed. It says nothing about *how* or *by whom*.
- **ML IDS alone:** models trained on public datasets score near 100% on that same dataset, but drop to about chance level on a different network (cross-dataset failure). Labelled data from your own network is rarely available.
- **Existing tools** (Wazuh FIM + Suricata, SIEMs) correlate host and network alerts by time window. They do not use cryptographic verification as ground truth, and they do not localize tampering to a path segment.

## 3. System architecture

```text
Sender ──► Relay 1 ──► Relay 2 ──► Receiver
  │   seg A   │   seg B   │   seg C   │
  │           │           │           │
  └─ hop-chained HMAC checked at every checkpoint ─┘
         │                │               │
     Zeek sensor A    Zeek sensor B   Zeek sensor C   (flow metadata per segment)
         │                │               │
         └──────────► Collector (FastAPI + DB, hash-chained event log)
                              │
                 ML model (XGBoost) ◄── weak labels from integrity failures
                              │
                     Streamlit dashboard
```

Components:

1. **Endpoint / checkpoint agents.** Each one computes SHA-256 and a hop-chained HMAC: `tag_i = HMAC(k_i, data || tag_{i-1})`. It verifies on receipt and sends a signed integrity event to the collector.
2. **Network sensors.** One Zeek sensor per segment produces flow metadata (duration, bytes, packets, flags, ports, timing). Payloads are not collected.
3. **Collector.** Stores integrity events and flows in a hash-chained (tamper-evident) log.
4. **ML model.** Baseline XGBoost / Random Forest trained on a public flow dataset, then adapted to the lab network (see Novelty A).
5. **Dashboard.** Four separate panels: **Integrity**, **ML risk**, **Tampered segment**, **Fused alert**.

## 4. Novelty

### A. Integrity-anchored weak supervision

- An HMAC failure is a positive label with almost no false positives. Key or configuration errors are logged separately and excluded.
- Flows on the failing segment, inside the failure's time window, are labelled *suspicious*. All other flows stay *unlabelled*.
- We fine-tune the public-dataset model on these labels with **PU learning** (positive-unlabelled). This adapts it to our own network without any manual labelling.
- **Closest prior work:** weak labelling of LDAP queries from nearby endpoint detections (arXiv 2606.28917). We use the same mechanism with a different label source: cryptographic verification, whose false-positive rate is known.

### B. Checkpoint-localized correlation

- The hop-chained HMAC tells us **which segment** the data changed on.
- Only that segment's flows are scored by the ML model. The integrity result (hard evidence) and the ML score (calibrated probability) are fused into one ranked alert.
- **Closest prior work:** path validation and traceback (OPT, SIGCOMM 2014; SPIE hash-based IP traceback). We found no work that couples these with an ML flow model.

> **Novelty claim scope:** web, arXiv and patent searches found no direct earlier statement of A or B. IEEE Xplore, ACM DL and Google Scholar were not searched directly, and no paper was read in full. We claim these as *variants / combinations*, not as "first".

## 5. Datasets

| Purpose | Dataset | Why | Caution |
|---|---|---|---|
| Base ML training | **CIC-IDS2017, corrected version** (Engelen et al., 2021) | Labelled benign and attack flows; the corrected release fixes the original's labelling and flow errors | Use the corrected release, not the original CSVs |
| Cross-dataset test | **CSE-CIC-IDS2018** (corrected by Liu, Engelen et al., 2022) | Similar attack classes; tests generalization | Expect a large drop; this is the problem Novelty A targets |
| Optional IoT extension | **TON_IoT** (UNSW) | Network + host traces with ground-truth timestamps | Free for academic use only |
| Integrity + localization evaluation | **Own lab dataset** (generated) | Public datasets have no file-tampering labels | Generated in the Docker lab below |

**Feature consistency rule:** train and test with the **same flow extractor**. CICFlowMeter features ≠ Zeek conn.log features. Either run CICFlowMeter on the lab pcaps, or retrain on Zeek features.

## 6. How we will build it

### Phase 1: Lab setup (Week 1)
- Docker Compose: `sender`, `relay1`, `relay2`, `receiver`, `attacker`, three Zeek sensors, `collector`.
- The attacker container runs an in-path proxy that alters payloads on one chosen segment, plus background noise traffic (scans, bursts).
- Isolated Docker network only. No real or unauthorized networks.

### Phase 2: Integrity layer (Week 1–2)
- Python agents: SHA-256, hop-chained HMAC, verification, signed events to the FastAPI collector.
- Keys come from environment variables or a secrets file. They never appear in code or logs.
- Hash-chained event log in SQLite or Postgres.

### Phase 3: Baseline ML (Week 2)
- Train XGBoost on corrected CIC-IDS2017 (binary: benign vs suspicious).
- Split **by day/session**, not by random rows.
- Test cross-dataset on CSE-CIC-IDS2018 to show the generalization gap.

### Phase 4: Novelty A (Week 3)
- Generate lab runs: normal transfers + tampering runs on random segments.
- Build weak labels from integrity failures (segment + time window).
- Fine-tune with PU learning and compare against the baseline on held-out lab sessions.

### Phase 5: Novelty B (Week 3)
- Localize tampering from the hop-chained HMAC.
- Score only that segment's flows and fuse the scores into a ranked alert.

### Phase 6: Dashboard + evaluation (Week 4)
- Streamlit dashboard with the four panels.
- Run all metrics and checks below.

## 7. Evaluation

| Component | Metrics |
|---|---|
| Integrity layer | Tamper-detection rate, false positives, verification latency |
| Baseline ML | Precision, recall, macro-F1, FPR, confusion matrix (within- and cross-dataset) |
| Novelty A | Macro-F1 on held-out lab sessions, **before vs after** weak-label fine-tuning |
| Novelty B | Segment-localization accuracy; irrelevant alerts shown per incident |
| System | Event ingestion delay, alert latency, inference time |

**Checks that must be able to fail:**
- **A:** shuffle the weak labels. The improvement must disappear; if it doesn't, the gain is not from the labels.
- **B:** a tamper on segment *k* must never be attributed to segment *j*. Test every segment.
- **Integrity:** flip one byte of the payload. Verification must fail.

## 8. Tech stack

Python, FastAPI, SQLite/Postgres, Zeek, CICFlowMeter, scikit-learn, XGBoost, Streamlit, Docker Compose.

## 9. Security and ethics

- Run only in the isolated lab network.
- Never hard-code or log keys.
- Store flow metadata only, no payloads. Set a retention limit.
- Public datasets are a starting point. Model results do not carry over to a real network without validating on that network.

## 10. Key references

- Engelen, Rimmer, Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE SPW 2021.
- Liu, Engelen et al. *Error Prevalence in NIDS datasets: CIC-IDS-2017 and CSE-CIC-IDS-2018.* IEEE CNS 2022.
- Cantone et al. *On the Cross-Dataset Generalization of ML for Network Intrusion Detection.* arXiv 2402.10974.
- Arp et al. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022.
- *ML-Powered LDAP Reconnaissance Detection using Weak Supervision.* arXiv 2606.28917.
- Kim et al. *Lightweight Source Authentication and Path Validation (OPT).* SIGCOMM 2014.
- Snoeren et al. *Hash-Based IP Traceback (SPIE).* SIGCOMM 2001.
- Crosby, Wallach. *Efficient Data Structures for Tamper-Evident Logging.* USENIX Security 2009.
- Skinner, Valdes. *Probabilistic Alert Correlation.* RAID 2001.
