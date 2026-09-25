# Submission log

| Version | File | Time (IST) | CV macro F0.5 | Public LB | What changed |
|---|---|---|---|---|---|
| v1 | `v1_matching_results.tsv` | 25 Sep 03:02 | 0.9753 (10% train slice, 3-fold) | not submitted (team decision: broken full-density shortlist) | TF-IDF name+address shortlist (top 10 each, same country), 40 features, LightGBM, each S2/S3 row to its best S1 if prob >= 0.75 |
| v2 | `submissions/v2/` | Fri 17:00 | 0.97904 (FULL OOF) | **0.9677** | two-stage LightGBM, word-level shortlist, expected-F0.5 decision (1633975 S1 rows with matches) |
| v3 | `submissions/v3/` | Fri 17:15 | 0.98015 (FULL OOF) | **0.9700** | v2 + sibling-agreement features in stage 2 (GPU XGBoost stage 2) (1634819 S1 rows with matches) |
| v5 | `submissions/v5/` | Fri 17:35 | 0.98015 (FULL OOF) | **0.838** -> France ~0.937, US+India ~0.976 | PROBE: v3 with all France rows emptied (measures France score) (1388739 S1 rows with matches) |
| v4a | `submissions/v4a/` | Fri 18:49 | 0.98005 (FULL OOF) | _upload & fill in_ | stage 1 as deeper GPU XGBoost (depth 10, 1000 rounds) + v3 stage 2 (1634872 S1 rows with matches) |
| v4 | `submissions/v4/` | Fri 19:56 | 0.97753 (test-like OOF, not comparable to FULL) | **0.961357** (-0.0086 vs v3: test-like data halved US candidate lists, 15.9 vs 31.3 on test) | trained on test-like conditions (US S1 at test density, look-alike records topped up) + sibling features, GPU XGBoost (1633498 S1 rows with matches) |
| v6a | `submissions/v6a/` | Fri 20:46 | 0.97966 (test-like OOF, calibrated) | NOT UPLOADED (built on v4's test-like models) | v4 + consensus stage-2 features + calibrated per-country decision (held-out 0.97966 vs v4 0.97753, test-like validation) + France rows from French-cleaned rebuild (1637900 S1 rows with matches) |
| v6 | `submissions/v6/` | Fri 21:32 | 0.98151 (FULL OOF, same validation as v3) | _upload & fill in_ | v3 base (full data) + consensus stage-2 features (held-out 0.98151 vs v3 0.98015) + France rows from French-cleaned rebuild; v3's decision rule, no calibration (1635015 S1 rows with matches) |
