# Verification report

| check | result | detail |
|---|---|---|
| unit: a one-byte payload flip fails verification | PASS | 1 passed, 114 deselected, 1 warning in 4.42s |
| unit: a tamper on segment k is attributed to k only | PASS | 15 passed, 100 deselected, 1 warning in 4.33s |
| unit: the event log detects a modified row | PASS | 1 passed, 114 deselected, 1 warning in 4.43s |
| unit: the weak-label builder excludes key/config errors | PASS | 1 passed, 114 deselected, 1 warning in 4.43s |
| leakage: no identifier/timestamp columns in the model feature set | PASS | 99 columns, offending: [] |
| leakage: no session appears in two splits | PASS | 10 splits checked, overlaps: [] |
| leakage: the held-out attack type is absent from train and val | PASS | leaks: [] |
| leakage: weak labels are identical with ground truth and plan removed from the manifest | PASS | 12 sessions |
| control: shuffled weak labels remove the gain (XGBoost/MLP naive, Elkan-Noto, MIL, nnPU) | PASS | worst arm real-shuffled gap 0.194; seeds where gain vanishes in the weakest arm: 5/5 |
| control: go/no-go (true-label model separates attack from benign) | PASS | PR-AUC 0.929 >= 0.8 |
| control: a model on segment id alone scores at chance | PASS | 0.105 vs chance 0.121 |
| control: no tamper is attributed to the wrong segment; no tamper call on a clean session | PASS | 52 tampered sessions, 0 misattributed messages, 0 false calls on clean sessions |
| reproducibility: re-running a logged Phase 4 cell from its config gives the same PR-AUC | PASS | logged 0.867848 vs rerun 0.867848 |
| full test suite | PASS | 115 passed, 1 warning in 10.94s |