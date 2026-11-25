# Two-minute demo script (for a professor)

Start: `docker compose --env-file .env -f lab/docker-compose.yml up -d`, then `streamlit run dashboard/app.py` (http://localhost:8501).
If Docker is unavailable, skip step 1 and use the simulator tab: everything else works offline.

| # | Click | Say |
|---|---|---|
| 1 | **Live demo** tab, segment **B**, attack **flip_byte**, **Run live attack** | "This runs a real session in an isolated Docker lab. A sender pushes hop-chained HMAC messages through two relays; an on-path attacker flips a byte on segment B. You can see each stage: capture, flow extraction, integrity verification, localization." |
| 2 | **Open it in the Session explorer** | "Four panels. Integrity: the HMAC chain failed on these messages. ML risk: a model flags flows on all segments. Tampered segment: cryptography says B. Fused alert: only B's flows." |
| 3 | Look at the **three alerting modes** table | "ML-only and time-window alerting raise irrelevant alerts. Localized fusion keeps the same recall but drops them. Across 5 seeds this is about 10x fewer irrelevant alerts per incident." |
| 4 | **Tampered segment** card -> **Explain** | "Why B and not C? Relays tag what they actually received, so after the tamper the next hop is consistent. Only the segment where the emitted digest differs from the received one is blamed. A wrong key is a separate verdict, `config_error`, never a tamper." |
| 5 | **Fused alert** card -> **Explain**, pick a flow | "Why this flow? Exact TreeSHAP shows which behaviours pushed the risk up, with plain-language names, for example variation in reply packet sizes." |
| 6 | Move the **counterfactual sliders** | "If I change this feature the model re-scores immediately. It shows what the model treats as suspicious." |
| 7 | Move the **threshold** and **window** sliders in the sidebar | "Everything recomputes live. A wider window catches more attack flows but also more benign ones." |
| 8 | **Trust & controls** tab -> **Re-train with shuffled labels now** | "The control: if the gain came from lab artefacts, random labels would work as well. They do not: PR-AUC falls to chance. The gain depends on the integrity-derived labels." |
| 9 | **Live demo** -> **Simulator**, change one letter / move the attacker | "Same integrity code, no Docker. The blamed segment always follows the attacker, never a later one." |
| 10 | **Limitations** tab | "Synthetic lab, enlarged attack footprint, 100 sessions. Novelty is a combination of prior ideas, and nothing here shows transfer to a real network." |
