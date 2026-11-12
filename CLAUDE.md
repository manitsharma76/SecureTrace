# SecureTrace

Research project: cryptographic integrity checks (hop-chained HMAC) on a multi-hop transfer path produce weak labels and a tampered-segment location, which are used to adapt and focus an ML network-intrusion model. ML is the core of the project; the crypto and Docker lab exist to produce data and labels.

- Idea, novelty claims, datasets, references: [SecureTrace_Project_Plan.md](SecureTrace_Project_Plan.md)
- Phases, design decisions, verification: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

## Status

Planning only. No code exists yet. The layout, stack and commands below are the agreed targets from DEVELOPMENT_PLAN.md; check that a file or command exists before relying on it, and update this file as phases land.

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
