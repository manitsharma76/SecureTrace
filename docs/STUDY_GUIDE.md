# SecureTrace study guide

Written for: the presenter, who needs to explain every part of the project and answer hard questions. Read sections 0 to 3 first, then the numbers sheet (section 12), then drill the Q&A bank (section 14). Every figure below comes from a logged run or a file in this repo; where something was *not* done or is weak, it says so, because that is what an examiner will probe.

---

## 0. The pitch

**30 seconds.** Cryptography can tell you *that* data was changed and *where* on a path, but nothing about the attack. A machine-learning intrusion detector can describe suspicious network behaviour, but fails on any network it wasn't trained on. SecureTrace connects them: a hop-chained HMAC on every message finds the tampered segment, and those integrity failures become free, trustworthy labels that adapt the ML model to our own network and tell it where to look.

**2 minutes.**
1. *Problem.* ML intrusion detectors trained on public datasets collapse on a new network (in our lab: PR-AUC about 0.2 against an attack rate of 0.175, i.e. chance). Labelled data from your own network almost never exists.
2. *Idea.* A failed integrity check is an almost-false-positive-free signal that something was tampered with, on a specific segment, in a specific time window. Use it as a weak label.
3. *What we built.* A Docker lab (sender, two relays, receiver, an on-path attacker), an integrity layer, a collector with a tamper-evident log, flow extraction, an ML pipeline and a dashboard.
4. *Result.* Adapting with integrity-derived labels lifts PR-AUC from about 0.2 to 0.8 to 0.9; shuffled labels erase the gain. Localizing to the failing segment cuts irrelevant alerts per incident about ten-fold at the same recall.
5. *Honesty.* Synthetic lab, enlarged attack footprint, 100 sessions; we claim variants and combinations of prior work, not firsts.

**10-minute structure.** Problem (1) -> architecture and lab (2) -> integrity layer with the one-byte-flip example (2) -> data and labels (1) -> results Q1 to Q4 (3) -> live demo (1 to 2) -> limits and claims (1).

---

## 1. Vocabulary (know these cold)

| Term | Meaning in this project |
|---|---|
| **Hop-chained HMAC** | Each node verifies the previous tag, then adds its own: `tag_i = HMAC-SHA256(k_i, SHA256(data) || tag_{i-1})`. Data is hashed first so the collector can re-verify from digests alone and payloads never leave the lab nodes. |
| **Segment** | A hop between two nodes: A = sender->relay1, B = relay1->relay2, C = relay2->receiver. Each has its own key k_A, k_B, k_C. |
| **Integrity event** | A signed report from a node to the collector: digest in/out, tag chain in/out, whether verification passed. Digests and tags only, no payload. |
| **Localization** | The collector compares what each segment's emitter sent with what its verifier received, and names the segment where they differ. |
| **Verdicts** | `ok`, `tamper`, `config_error` (keys/config disagree, not an attack), `incomplete` (missing events). |
| **Weak label** | Bag-level: a flow on the located segment inside the failure window is *suspicious*. Everything else is *unlabelled* (not negative). |
| **Bag** | (segment, time window) opened by an integrity failure. |
| **True label** | The attacker's own per-flow log. **Evaluation only, never used for training.** |
| **PU learning** | Positive-Unlabelled learning: you have some confirmed positives and a pile of unlabelled data. |
| **MIL** | Multiple-instance learning: a label applies to a bag, not to each item in it. |
| **PR-AUC** | Area under the precision-recall curve; the right metric when attacks are rare. Chance level equals the attack rate (about 0.175 here). |
| **FPR at 90% recall** | False-positive rate when the threshold is set to catch 90% of attacks. |
| **ECE** | Expected calibration error: average gap between predicted probability and observed frequency. |
| **TreeSHAP** | Exact additive feature contributions for tree models; we use XGBoost's built-in `pred_contribs`. |
| **Held-out attack type** | One mutator (`flip_byte` or `rewrite_size`) is excluded from training/validation and appears only in test. |
| **Novelty A / B** | A: integrity-anchored weak supervision. B: checkpoint-localized correlation and fusion. |

---

## 2. System architecture and the lab

**Path:** `sender --A--> relay1 --B--> relay2 --C--> receiver`, with a collector and sensors beside it.

**The attacker is genuinely on-path.** Each logical segment is split into two point-to-point Docker networks (`segX_in` and `segX_out`, subnets 172.28.11/12 for A, 21/22 for B, 31/32 for C, control network 172.28.9) and the attacker container bridges them. Its TCP-terminating relay (`lab/attacker/proxy.py`, transfer port **9100**) reads the message, may mutate it, and forwards it. This replaced an ARP-spoofing design; the topology gives a real man-in-the-middle without off-path tricks.

**The attacker runs on all three segments in every session** (pass-through on the untargeted ones), because it sits on the wire: without it a segment carries no traffic. It holds **no keys** (each node only gets its own two segment keys).

**Attack mutators**
- `flip_byte`: changes one payload byte.
- `rewrite_size`: truncates to 10 to 40% of the message or pads by 1 to 4x.
- The tampered segment also re-segments traffic into 512-byte chunks with 0.01 s pacing; pass-through segments are near-instant.
- Optional **recon** step: benign-shaped probing of the other segments for about 3 s.

**Benign background traffic:** HTTP (port 8080), an interactive service (2222) and UDP (5353), generated on every segment on both sides of the attacker.

**Sensors:** `tcpdump` on each segment's two bridge interfaces (found by their fixed subnets, since Docker's `br-<hash>` names carry no segment info), merged into one pcap per segment, then **CICFlowMeter** (a vendored, corrected fork) extracts flows. Zeek's `conn.log` is also produced but is not used by the model or shown in the dashboard. NIC offloads (TSO/GSO/GRO/LRO) are disabled so pcaps hold real wire-size packets.

**Collector:** FastAPI + SQLite, port 9000 on 127.0.0.1. It verifies each event's signature and appends it to a **hash-chained log**.

**Keys and secrets:** from environment variables only (`LAB_KEY_A/B/C`, at least 16 bytes; placeholders such as "replace-me" are rejected). Each node's event-signing key is derived from a master secret with `HMAC(secret, "securetrace/event/" + node)`, so a node cannot sign as another node. Keys never appear in code, logs or configs.

---

## 3. The integrity layer in detail

**Worked example (say this out loud).** Sender hashes the message: `d = SHA256(data)`, computes `t_A = HMAC(k_A, d)` and sends `data, [t_A]`. Relay1 recomputes `d`, checks the newest tag against k_A, then (always, even if the check failed) extends the chain with `t_B = HMAC(k_B, d || t_A)` over the data it actually received and forwards it. Relay2 and the receiver do the same with k_B and k_C.

Now an attacker flips one byte on segment B. Relay2 receives different data, so its recomputed digest differs from the one relay1 sent: `digest_changed = true` on B, and verification fails at relay2. Relay2 still tags *the data it received*, so the hand-off on segment C is internally consistent and verifies. Only B shows a difference, so only B is blamed.

**Localization rule (from `securetrace/integrity/localize.py`).** For each segment compare the emitter's output with the verifier's input:
- digest or tag changed -> **tamper** on that segment;
- hand-off identical but verification failed, or the emitter's tag is not what the collector's key reproduces -> **config_error** (never a tamper);
- missing events -> **incomplete**.

**Why relays forward what they received.** If they dropped or re-signed differently, one tamper would cascade into false failures on every later segment and localization would break.

**Why an altered older tag is still caught.** Each tag covers the previous tag, so editing an earlier tag makes the later verification fail. The test suite covers this explicitly.

**Tamper-evident log.** Each row stores a hash of its content plus the previous row's hash. The tests show it detects a modified row, a deleted or inserted row, a reordered row, an edited index, and tail truncation when the head is anchored.

**Tests that back this (all in `tests/test_integrity.py`):** one-byte flip fails verification; a tamper on segment k is attributed to k only for all three segments and five attack types; wrong or missing key = config error not tamper; missing events = incomplete; placeholder/short keys rejected without leaking them; event signature detects an edit and cross-node forgery.

---

## 4. Lab data generation

`scripts/generate_lab_runs.py` drives real randomized sessions through `docker compose exec`: picks a tamper segment (or none, about 60% tampered), a mutator, whether recon precedes it, 8 to 25 messages, message gaps and noise duration; starts the attacker relays, capture and noise; runs the sender; stops capture; extracts flows; fetches the collector's localization; and writes one JSON manifest per session joining
- `localizations` (what the collector concluded: the **weak-label source**),
- `ground_truth` (the attacker's per-message log: the **true label**, evaluation only),
- `log_integrity` (hash-chain verification).

Tamper segment and mutator are cycled evenly (`--tamper-offset` continues the cycle) because random choice skewed heavily toward one segment in small samples, which let a segment-identity-only model beat chance.

**True label definition:** a flow is truly malicious if it carries transfer traffic (an endpoint on port 9100) and starts within 1 s of a tampered message the attacker relayed on that segment. Requiring port 9100 keeps concurrent benign noise benign.

**Final dataset:** 100 sessions (52 tampered, 48 clean), about 6,100 test flows, attack rate 0.175.

---

## 5. The machine-learning pipeline

### 5.1 Phase 1: public data and the cross-dataset gap (Q1)
- **Data:** corrected CIC-IDS2017 (Engelen et al., 2021) and corrected InSDN, both from one Zenodo record so the *same corrected CICFlowMeter* is used throughout. Train Mon to Wed, tune the threshold on Thursday, test within-dataset on Friday, test cross-dataset on all of InSDN. Split by day, never by row.
- **Why not CSE-CIC-IDS2018?** No corrected CSV is published, and its original CICFlowMeter output would mix extractors. It is kept only as a legacy sensitivity config.
- **Features:** 99 shared CICFlowMeter columns; identifiers (IPs, ports, flow id, timestamps, protocol) are dropped.
- **Models:** logistic regression, random forest, XGBoost, LightGBM, a plain MLP; 5 seeds; training capped at 300,000 rows.
- **Results (mean over 5 seeds):**

| Model | Within 2017: PR-AUC / macro-F1 | Cross InSDN: PR-AUC / macro-F1 | FPR at 90% recall within -> cross |
|---|---|---|---|
| XGBoost | 0.994 / 0.953 | 0.902 / 0.673 | 0.000 -> 0.335 |
| Random forest | 0.995 / 0.987 | 0.828 / 0.624 | 0.000 -> 1.000 |
| MLP | 0.976 / 0.950 | 0.741 / 0.354 | 0.017 -> 0.926 |
| Logistic regression | 0.946 / 0.920 | 0.941 / 0.802 | 0.038 -> 0.288 |

**Be honest about this table:** cross-dataset PR-AUC on InSDN stays fairly high; the gap shows mainly in macro-F1 and in the false-positive rate at fixed recall (the decision threshold does not transfer). The dramatic collapse to chance is on *our lab*, section 7.

### 5.2 One flow extractor
Model input is always corrected-CICFlowMeter features, for public and lab data alike. This is a hard rule because CICFlowMeter and Zeek features differ.

### 5.3 Splitting and leakage rules
Split by day (public) or by **session** (lab); no session in two splits (10 splits checked). One mutator is held out of train and validation. Identity features (IPs, ports, segment, timestamps) never enter the model. True labels never train anything.

---

## 6. The go/no-go story (a good example of honest science)

The plan said: before any weak-label work, a model trained on the **true** lab labels must be able to separate attack from benign. If not, the attacks leave no footprint and the whole idea cannot be tested.

| Attempt | Result |
|---|---|
| First attacker (small size changes), 25 sessions | **NO-GO:** held-out PR-AUC 0.213 vs about 0.20 chance (in-sample 0.76 to 0.83 = overfitting) |
| Bigger footprint (`rewrite_size` 10 to 40% / 1 to 4x, 512-byte re-segmentation, pacing only on the tampered segment) | **GO:** 0.917 on 33 sessions |
| First 113-session dataset | 0.934, **but it contained no benign traffic at all** (noise clients targeted addresses the sender could not route to), so every bag was trivially pure and the whole comparison would have been meaningless. Discovered, fixed, regenerated. |
| Final 100 sessions with benign noise | **0.929**; segment-identity-only control 0.105 vs about 0.121 chance |

**Take-away to say:** the footprint was deliberately enlarged to be detectable, so detectability is by construction; `flip_byte` alone leaves little flow-level trace.

---

## 7. Weak labels and the "before" numbers (Phase 3)

**Bag construction.** Each `tamper` message gives (segment, time). Nearby same-segment events are merged; the bag is padded by half a window each side. A flow overlapping a bag on its segment is a weak positive; everything else is unlabelled. `config_error` and `incomplete` never create bags.

**Label quality vs window width** (against the true labels):

| Window | 0.5 s | 1 s | 2 s | 5 s | 10 s |
|---|---|---|---|---|---|
| Bag precision | 0.87 | 0.84 | 0.80 | 0.73 | 0.68 |
| Bag recall | 0.98 | 0.99 | 0.99 | 0.996 | 0.997 |

Wider window = more attack flows caught but more benign ones swept in.

**"Before" (no adaptation), PR-AUC, `flip_byte` / `rewrite_size` held out:** MLP 0.335 / 0.332; LightGBM 0.27 / 0.25; XGBoost 0.217; logistic regression 0.21; random forest 0.17. Chance is 0.175.

**Splits:** 44 train, 15 validation, 41 test sessions (all sessions of the held-out mutator go to test, plus a share of clean ones).

---

## 8. Novelty A: weak-label adaptation (Phase 4)

**Methods (all start from the public model of the same seed)**
- **Naive:** unlabelled treated as negative (the baseline to beat).
- **Elkan-Noto PU (XGBoost):** train g(x) = P(labelled | x) out-of-fold by session; label frequency c = mean g over held-out labelled positives; each unlabelled flow enters twice, as positive with weight w = (1-c)/c * g/(1-g) and as negative with 1-w; continue boosting from the public model.
- **nnPU (MLP):** non-negative PU risk with a sigmoid loss and a class prior pi = P(s=1)/c.
- **Attention-MIL (MLP):** a bag-level nnPU risk over attention-pooled instance scores; per-flow scores read from the instance head.

**Results, PR-AUC (`flip_byte` / `rewrite_size` held out):**

| Method | PR-AUC | Gain over before |
|---|---|---|
| XGBoost naive | 0.895 / 0.795 | +0.68 / +0.58 |
| XGBoost Elkan-Noto | 0.879 / 0.796 | +0.66 / +0.58 |
| MLP naive | 0.819 / 0.851 | +0.48 / +0.52 |
| MLP attention-MIL | 0.686 / 0.638 | +0.35 / +0.31 |
| MLP nnPU (estimated prior) | 0.404 / 0.393 | +0.07 / +0.06 |
| MLP nnPU, prior = labelled fraction (post-hoc ablation) | 0.596 / 0.562 | +0.26 / +0.23 |

**The control that can fail:** shuffle the weak labels and the gain must vanish. It does, in **all 14 arms over all 5 seeds** (XGBoost shuffled about 0.16 to 0.22, which is chance).

**What to say about naive winning.** The weak labels are nearly complete (recall about 0.99) in this lab, so "unlabelled = negative" is almost true; the extra machinery of PU/MIL only adds variance. This does *not* undermine Novelty A: A is about the **source** of the labels (cryptographic verification), not the algorithm. PU/MIL matter when labels are incomplete.

**Why nnPU is poor.** The Elkan-Noto estimate gives c about 0.62 to 0.65 (prior about 0.21) while true label recall is about 0.98 and the real malicious fraction about 0.11. A synthetic test shows the estimator is **biased low whenever positives overlap other flows**, which inflates the prior; nnPU then over-predicts positives. Hyper-parameters were fixed before looking at test scores and not tuned; the labelled-fraction prior variant is a *post-hoc* ablation and is labelled that way.

**Sweeps.** *Window width:* XGBoost is robust (0.86 to 0.88 for `flip_byte` across 0.5 to 10 s); MIL/nnPU degrade as bag precision falls. *Label efficiency (share of tampered training sessions):* XGBoost reaches 0.73 / 0.63 at 25% (about 13 incidents) and saturates by 50% (about 26): 0.89 / 0.83. nnPU gets *worse* with more data, consistent with the prior problem above.

---

## 9. Label-free comparators (Phase 5, Q3)

All use the lab TRAIN flows **without labels**, on the same split and seeds.

| Method | PR-AUC (`flip_byte` / `rewrite_size`) |
|---|---|
| Isolation forest | 0.21 / 0.25 |
| Autoencoder | 0.23 / 0.39 |
| CORAL (align covariances, public -> lab) | 0.20 / 0.21 |
| DANN (adversarial domain adaptation) | 0.45 / 0.31, 95% CI about +/-0.25 (very unstable) |
| SCARF encoder pretrained on lab flows + public-data head | 0.13 / 0.15 (below chance) |
| **Weak-label adaptation (for comparison)** | **0.8 to 0.9** |

**Best-combination test.** SCARF pretraining followed by weak-label fine-tuning scored 0.814 / 0.847, versus the same network from random initialisation 0.810 / 0.864: a paired difference of +0.003 +/- 0.018 and -0.017 +/- 0.014. Self-supervised pretraining contributes nothing measurable, and the from-scratch network matches the public-initialised one, so public pretraining is not what drives the gain once weak labels exist.

**Caveats to volunteer:** anomaly detectors were fit on lab flows contaminated with attacks; comparators were not hyper-parameter tuned; the optional GRU/transformer sequence-model stretch was not done.

---

## 10. Novelty B: localization and fusion (Phase 6, Q4)

**Alerting modes compared on each test incident** (a test session where the collector flagged a tamper; 31 per split):
1. **ML only** over all segments.
2. **Time-window correlation** (how a SIEM works): any flow on **any** segment overlapping the integrity-alert window.
3. **Localized fusion** (ours): only flows on the **located** segment inside its window, ranked by calibrated risk, with the integrity evidence attached.

**Results (alert = calibrated risk >= 0.5):**

| | Irrelevant alerts per incident | Recall |
|---|---|---|
| ML only | 1.27 / 0.99 | 0.69 / 0.67 |
| Time-window | 1.23 / 0.94 | 0.68 / 0.66 |
| **Localized fusion** | **0.10 / 0.10** | 0.68 / 0.66 |

About ten times fewer irrelevant alerts at the same recall. On *clean* sessions localized and time-window raise nothing (no integrity failure), versus 0.14 / 0.10 false alerts per clean session for ML-only.

**Localization accuracy:** 52 of 52 tampered sessions attributed to the true segment, 0 misattributions, 0 false tamper calls on clean sessions. (A handful of individual tampered messages come back `incomplete`, which loses a little recall but never mislabels a segment.)

**Do not claim precision@k.** It is about 1.0 in every mode (a ceiling effect: inside an incident session the model already ranks real attack flows first), so it does not support Novelty B.

**Calibration.** Isotonic and temperature calibrators are fit on **weak validation labels**. ECE against true labels: raw 0.028 / 0.049, isotonic 0.017 / 0.033, temperature 0.020 / 0.054. A modest isotonic improvement whose intervals overlap; in the earlier 40-session run calibration made things *worse*. Say "unproven", not "calibrated".

**Explanations.** Exact TreeSHAP per flow (XGBoost `pred_contribs`), shown with plain-language feature names and the integrity evidence.

**A mistake we caught and fixed (good story):** the first time-window baseline reused the localized segment's flows, so it matched the localized mode exactly and understated the baseline's weakness. It now takes flows on any segment overlapping the alert time.

---

## 11. The dashboard and demo

Tabs: **Session explorer**, **Live demo**, **Trust & controls**, **How it works**, **Limitations**.
- Four clickable cards: Integrity, ML risk, Tampered segment, Fused alert.
- Sliders for alert threshold and weak-label window; everything recomputes live.
- Three-mode comparison table and chart for the selected session.
- **Hop-chain diagram** (why this segment), with live digests/tags from the collector when the lab is up.
- **Per-flow TreeSHAP bars** and **counterfactual sliders** ("what changes if...?").
- **Trust tab:** reliability plot + ECE, label-quality vs window curve, shuffled-label control (stored results and a live re-train button, about a minute).
- **Live demo:** real attack in Docker (about 1 to 2 minutes, streamed step by step) or an offline **simulator** on the real integrity code with fresh random keys.
- Columns marked "evaluation only" use the attacker's log and never feed a score.
- The two-minute click script is in `docs/DEMO_SCRIPT.md`.

---

## 12. Numbers cheat sheet

| Fact | Value |
|---|---|
| Lab sessions | 100 (52 tampered, 48 clean); splits 44 / 15 / 41 sessions |
| Test flows / attack rate | about 6,100 / 0.175 |
| Public model on lab (before) | PR-AUC 0.17 to 0.34 (chance 0.175) |
| True-label go/no-go ceiling | 0.929 (threshold 0.8); segment-only control 0.105 vs 0.121 |
| Weak-label bag precision / recall at 2 s | 0.80 / 0.99 |
| Weak-label adaptation, XGBoost | 0.895 / 0.795 (+0.68 / +0.58) |
| Shuffled labels | about 0.16 to 0.22 = chance; 14 of 14 arms |
| Label-free best | DANN 0.45 / 0.31, unstable |
| Localization | 52 / 52 correct, 0 misattributed |
| Irrelevant alerts per incident | 1.27 / 0.99 (ML-only) vs 0.10 (localized) |
| ECE raw -> isotonic | 0.028 -> 0.017 and 0.049 -> 0.033 |
| Seeds | 5 everywhere; fixed config + MLflow log per run |
| Tests / verification | 146 tests; 14 / 14 verification checks; end-to-end check passes |
| Time to generate a session | about 2 minutes |

---

## 13. Limitations and what changed (say these before they are asked)

1. **Synthetic lab.** Traffic, noise and the attacker are ours. Train and test share the same lab, attacker implementation and noise generator. **Nothing here shows transfer to a real network.**
2. **Enlarged footprint.** The first attacker was undetectable at flow level. `rewrite_size` was deliberately enlarged; detectability is by construction. Only two mutators exist, so "held-out attack type" means one of two.
3. **Small data.** 100 sessions, about 31 incidents per split. Several 40-session conclusions changed at 100 (MIL improved, nnPU worsened, the ML-only alert count fell from about 10 to about 1), so expect further movement.
4. **Confidence intervals understate uncertainty.** The 5 seeds vary the model and adaptation, but the **session split is fixed (split seed 0)**. Variance from drawing a different train/test split is not included.
5. **No manual-label baseline.** We compare against label-free methods and a true-label ceiling (the go/no-go model, a different protocol), not against paying for hand labels.
6. **Weak-label noise.** Bag precision is 0.68 to 0.87; recall about 0.99; a few tampered messages are `incomplete`.
7. **Threats not covered.** Passive eavesdropping and reconnaissance leave no integrity failure, so they stay unlabelled. A compromised node that holds its own keys can forge its segment. Replays are not addressed (no nonce binding in the chain). Key distribution and rotation are out of scope.
8. **System performance not evaluated.** The original plan listed verification latency, ingestion delay and alert latency; we did not measure them.
9. **Not done:** the optional sequence-model (GRU/transformer) stretch. Zeek output is produced but unused by the model and the dashboard.
10. **Calibration is unproven** and uses noisy weak labels.

---

## 14. Question bank

**Concept**

*Q: Why not just use a normal IDS?*
A: It was trained elsewhere and fails on a new network (here: chance). We need labels from *our* network, and nobody has time to hand-label them. The integrity layer produces trustworthy labels for free.

*Q: Why not just use the HMAC failure as the alert?*
A: It proves a change but says nothing about how or by whom, and it only covers the data path being checked. The flows tell you what network behaviour accompanied it. The two are complementary: crypto gives where, ML gives what.

*Q: How is this different from a SIEM that correlates alerts by time?*
A: A SIEM correlates by time, which says *when* but not *where*. Our time-window baseline does exactly that and raises about as many irrelevant alerts as ML alone. Localization to a segment is what removes them (about ten-fold fewer).

*Q: Why are integrity failures good labels?*
A: A verified-wrong HMAC essentially cannot happen without a change, so false positives are nearly nil. Key and configuration problems are classified separately (`config_error`) and excluded so a misconfiguration cannot masquerade as an attack.

*Q: Why bag labels instead of per-flow labels?*
A: The integrity check says "something on segment B between t1 and t2 was tampered with", not which flow. That is a bag.

**Cryptography**

*Q: Why hash first and then HMAC?*
A: So the collector can recompute the whole chain from digests, and no payload ever leaves the nodes (privacy rule: flow metadata only).

*Q: Why do relays tag data that failed verification?*
A: So one tamper does not make every later segment fail. Each segment is judged by whether *its* emitted digest/tag matches *its* received one.

*Q: What stops a node from forging events?*
A: Per-node event keys derived from a master secret; the collector checks each signature. The tests include cross-node forgery.

*Q: What if the attacker steals a relay's keys?*
A: A node with its own keys can forge tags for its own two segments, so it can hide tampering that it performs itself. We assume honest nodes holding their keys; this is a stated limitation. The attacker container in the lab holds no keys.

*Q: What about replay?*
A: Not covered. The chain binds the data digest and previous tags, not a nonce or timestamp.

*Q: Is the log really tamper-evident?*
A: Rows are hash-chained. Modified, deleted, inserted and reordered rows are detected; truncating the tail needs an anchored head to detect (tested).

**Data and labels**

*Q: Isn't this circular: you label with integrity failures and then test on the attacker's logs?*
A: No. Training labels come from the collector's cryptographic verdicts. Test labels come from a separate source, the attacker's own per-flow log. A test confirms the weak labels do not change when ground truth is removed from the manifest.

*Q: Could the model just learn "segment B is bad"?*
A: We control for it: a model on segment identity alone scores at chance (0.105 vs 0.121), identifiers are dropped, and the tampered segment/mutator are cycled evenly.

*Q: Could the model just learn this one attacker's footprint?*
A: Partly, yes, and we say so. One of two mutators is held out and still scored well, which suggests some generalization between them, but with two mutators in one lab this is limited evidence.

*Q: Why did you enlarge the attack footprint? Isn't that cheating?*
A: The original attack left no flow-level trace (PR-AUC at chance even with true labels). A learned detector cannot detect what leaves no trace, so we enlarged the footprint to make the experiment meaningful, and we disclose it. It limits what can be claimed about subtle attacks.

*Q: Your first dataset had no benign traffic. How do you know the current one is right?*
A: We noticed that bag precision was exactly 1.0 for every window, which is suspicious, traced it to noise clients unable to reach their servers, fixed the topology and regenerated. Now bag precision is 0.68 to 0.87 and the capture contains ports 8080, 2222 and 5353.

**Machine learning**

*Q: Why PR-AUC?*
A: Attacks are rare (17.5%), so accuracy is misleading; PR-AUC compares against a chance level equal to the attack rate. We also report FPR at 90% recall.

*Q: Why XGBoost?*
A: Strong on tabular flow features; it is also the model we explain with exact TreeSHAP. We also trained logistic regression, random forest, LightGBM and an MLP.

*Q: Why does naive beat PU and MIL?*
A: Labels are almost complete here (recall about 0.99), so the "unlabelled = negative" assumption is nearly true; PU/MIL add variance without gaining. In a setting with missing labels they could matter. We report it as a result about this lab.

*Q: Why did nnPU do badly?*
A: The Elkan-Noto prior estimate is biased high here (about 0.21 vs a true malicious fraction of about 0.11), because the estimator is biased low whenever positives overlap other flows. The labelled-fraction prior fixed part of it, but that is a post-hoc ablation.

*Q: Did you tune on the test set?*
A: No. Hyper-parameters were fixed beforehand. I looked at test scores while debugging nnPU, so I did not select hyper-parameters from them, and the only post-hoc addition (the labelled-fraction prior) is labelled as such.

*Q: Does shuffling labels really prove anything?*
A: It shows the gain depends on *which* flows are labelled positive, not on the training procedure or lab artefacts: random labels give chance performance. It is a necessary check, not proof that the labels are good (label quality is measured separately).

*Q: How are your confidence intervals computed, and are they trustworthy?*
A: 95% t-intervals over 5 seeds. They do not include variance from the session split (fixed), so they are optimistic. Say so.

*Q: What is the upper bound?*
A: The go/no-go model trained on true labels gets 0.929 PR-AUC (a different protocol, so only roughly comparable to the weak-label 0.8 to 0.9).

*Q: Why calibrate at all?*
A: To fuse a model score with hard integrity evidence you want something probability-like. We found the benefit modest and unproven.

*Q: Why is the SSL result negative?*
A: With 44 training sessions and about 6,000 flows in total, contrastive pretraining has little to learn that the weak labels do not already supply, and the from-scratch network matches the public-initialised one.

**Localization and fusion**

*Q: Is 100% localization accuracy meaningful?*
A: It is a result for this lab with honest nodes and a single tampered segment per session. Individual tampered messages can come back `incomplete` (missing events), but none was ever attributed to the wrong segment.

*Q: What if two segments are tampered at once?*
A: The localizer returns all segments where the hand-off differs, and tests cover single-segment tampers; multi-segment attacks were not experimentally evaluated.

*Q: Why is precision@k not reported as a win?*
A: It is 1.0 in every mode because the model already ranks real attacks first inside an incident. The benefit is in alert counts (irrelevant alerts), not ranking.

**Practical**

*Q: Would this work on a real network?*
A: Unknown. Lab traffic is synthetic, the attacker is ours and train/test share one lab. It needs validation on a real network before any such claim.

*Q: What does it cost at runtime?*
A: Not measured (verification latency and alert latency were in the plan but not evaluated).

*Q: Is anything novel?*
A: We claim variants and combinations only. A is close to weak labelling of LDAP queries from nearby endpoint detections (arXiv 2606.28917) with a cryptographic label source. B is close to path validation and traceback (OPT, SPIE) coupled with a flow model. Our searches did not find a direct earlier statement, but IEEE Xplore, ACM DL and Google Scholar were not searched directly and no paper was read in full.

**Process**

*Q: How do I know the results are reproducible?*
A: Every experiment has a config and a fixed seed and is logged to MLflow; `python scripts/run_all.py` reruns everything; `python scripts/verify_project.py` re-runs a logged cell from its config and gets the same PR-AUC to six decimals.

*Q: How was the project tested?*
A: 146 automated tests (integrity, localization, weak labels, splits, adaptation, calibration, fusion, simulator, dashboard UI), 14 verification checks (including leakage and control checks that can fail), and an end-to-end run that drives a real session into the dashboard.

---

## 15. Things not to say

- "The first system to ..." (we claim variants and combinations).
- "This works on real networks" (synthetic lab only).
- "PU learning improves detection" (naive was as good or better here).
- "Calibration improves reliability" (modest, unproven).
- "Precision@k shows localization helps" (ceiling effect).
- "100% detection" (the model's recall at threshold 0.5 is about 0.66 to 0.69; localization accuracy is 100% on this lab).
- "We evaluated performance/latency" (we did not).

---

## 16. Demo checklist

1. `docker compose --env-file .env -f lab/docker-compose.yml up -d` (wait for all 7 services).
2. `streamlit run dashboard/app.py` -> http://localhost:8501; sidebar should say the collector is **live**.
3. Start the live attack **before** you start talking (it takes 1 to 2 minutes), or use the simulator tab as a no-Docker fallback.
4. Follow `docs/DEMO_SCRIPT.md`.
5. If something fails: the Session explorer works on saved sessions without Docker; the simulator needs nothing.

## 17. Where to find things

| Question | File |
|---|---|
| Results and caveats | `FINAL_REPORT.md`, `CLAUDE.md` |
| Plan and decisions | `DEVELOPMENT_PLAN.md`, `SecureTrace_Project_Plan.md` |
| Integrity code | `securetrace/integrity/` (`chain.py`, `localize.py`, `eventlog.py`) |
| Attacker | `lab/attacker/proxy.py` |
| Weak labels | `securetrace/data/weak_labels.py` |
| Adaptation | `securetrace/adapt/` |
| Fusion and explanations | `securetrace/fusion/` |
| Experiments | `scripts/` and `configs/` |
| Verification | `scripts/verify_project.py`, `scripts/verify_end_to_end.py`, `docs/verification_report.md` |
