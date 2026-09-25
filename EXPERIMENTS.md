# Experiment log

Every run: what changed, the score on held-out training data (CV), and the decision taken.
CV = macro F0.5 on out-of-fold predictions. Never compare scores from different validation setups without saying so.

## Validation setups
- **S10**: 10% of train S1 rows (chosen by id hash) + all their true matches + 10% of unmatched S2/S3 rows. 3 folds grouped by true S1. Fewer same-name competitors than test, so scores are optimistic.
- **FULL**: all 2.2M train S1 rows and all 10.3M S2/S3 rows (same density as test). Introduced in exp 3.

## Runs

| # | Date (IST) | Setup | Change | Recall ceiling | CV F0.5 | Decision |
|---|---|---|---|---|---|---|
| 0 | 25 Sep 01:30 | S10 | Rule only: best (name token-set + address token-set)/200, threshold 0.8 | 0.9870 | 0.8925 | Baseline reference |
| 1 | 25 Sep 01:30 | S10 | LightGBM, 40 features, TF-IDF max_df 5%/2% (relative) | 0.9870 | **0.9786** | Good, but test run stalled on US: relative max_df lets common 3-grams hit ~30k S1 rows each |
| 2 | 25 Sep 02:00 | S10 | TF-IDF max_df absolute 4000/2000 docs (speed fix) | 0.9869 | 0.9753 | Test run 65 min. **Submitted file v1.** Lost 0.003 because the cosine features also lost common 3-grams |

## Error analysis of run 2 (S10 OOF, 25 Sep 04:00) — `src/error_analysis.py`, log `work/errors_v1.log`
Score 0.9771 at t=0.75 (re-trained OOF). Points lost if each mistake type were fixed:

| Mistake type | Count | Points | Over-represented in (share vs share of all true pairs) |
|---|---|---|---|
| wrong_s1 (top candidate is another S1) | 16,512 | +0.0098 | non-Latin name 39% vs 7%; missing address 40% vs 4%; India 66% vs 40% |
| under_thresh (right S1, prob < t) | 16,735 | +0.0078 | missing address 21% vs 4%; non-Latin 16% vs 7% |
| decoy_merge (no-match row merged) | 4,883 | +0.0058 | near-identical name + slightly different house number (1229 vs 1224); same address + unrelated domain name (p=0.94) |
| blocking_miss (true pair not shortlisted) | 9,972 | +0.0054 | non-Latin name 60% vs 7%; missing address 19% vs 4% |

Decisions taken from this:
1. Non-Latin names are the biggest single cause -> fine-tuned multilingual embedding search + embedding-similarity feature (GPU).
2. No feature says how far the top candidate is ahead of the 2nd -> add margin-to-second features and "how many candidates look like this name" counts.
3. Missing-address rows only get name candidates -> larger name top-k for them; separate thresholds by segment.
4. Decoys differ in house numbers -> richer number features (alphanumeric tokens like "8-9-1/14a", digit edit distance).
5. Siblings (other S2/S3 rows already given to the same S1) carry the address a row may lack -> stage-2 features using the best sibling.

## Embedding search for non-Latin names (`src/embed.py`, GPU RTX 3050 4 GB)
Metric: recall@10 = true S1 among the 10 nearest of all 883k train India S1 rows, on 20k held-out non-Latin S2/S3 rows
(their S1 is outside the fine-tuning partition).

| # | Model / text | recall@10 | Decision |
|---|---|---|---|
| E0 | multilingual-e5-small, not fine-tuned, name only | 0.156 | Off-the-shelf model does not bridge scripts |
| E1 | fine-tuned (276k pairs, 1 epoch, in-batch contrastive, word-embedding table frozen for 4 GB), name only | 0.573 | Better, but repeated names (chains) cap name-only search |
| E2 | fine-tuned, text = name + " | " + cleaned address (64 tokens, gradient checkpointing) | **0.998** | **Use E2** for blocking + `emb_cos` feature |

GPU lessons: bs 256 with trainable word table OOMs; freeze word table; score queries against 883k S1 in batches of 256 (2048 needs 3.4 GB).

## v2 pipeline on S10 (25 Sep 05:10–06:10)
Eval = all S10 S1 rows (they are all outside the embedding fine-tuning partition, so no leakage). v1 on the same rows: 0.9771.

| # | Change | Recall ceiling | CV F0.5 | Decision |
|---|---|---|---|---|
| 3 | v2 stage 1: + embedding search/feature, full-vocab cosines, margin-to-2nd, ambiguity counts, alnum house-number features, name top-30 when no address; LightGBM 255 leaves, 600 rounds, 16M-row sample | **0.9961** (was 0.9869) | **0.9871** | +0.010 over v1. Keep all |
| 4 | same, sample rate 0.4 (8M rows, to fit RAM) | 0.9961 | 0.9863 | Less data costs 0.0008 -> full data should help |
| 5 | + stage 2 (S1-side competition: p1 margin, other rows' p1 for same S1, same-source counts) | 0.9961 | **0.9877** | +0.0014. Keep. Per-segment thresholds: +0.0000 -> turned off (slow) |

Error analysis of run 5 (`work/s10_errors.log`), points lost: wrong_s1 0.0045 (73% have NO address; chain names like "Perfect Food Pvt Ltd" at several S1 addresses), under_thresh 0.0043 (34% no address), decoy_merge 0.0036 (look like true matches: "Consolidated Médical Studios | 31 Laurel Circle" vs S1 "... | 315 Laurel Circle", p=0.99; likely irreducible), blocking_miss 0.0014 (31% no address). Non-Latin names are no longer a problem (0.2% of misses vs 60% in v1).

## FULL setup (all 2.2M train S1, 10.3M S2/S3; eval = 1.1M S1 outside the embedding partition)
Build: 42 chunks, ~100–130 s each (~85 min), 9.1 GB of features on disk.

| # | Change | Stage-1 sample | Result | Decision |
|---|---|---|---|---|
| 6 | Record-level 6% sample (SAMPLE_RATE) | 7.2M rows, only **182k positives** | stopped before training | Record sampling throws away 94% of positives. Switched to row-level sampling |
| 7 | Row-level: all eligible positives + 15% hard negatives (rank ≤ 2 by name cosine or address token-set, or name token-set ≥ 80) + 1.5% other negatives; stage 2 on 50% of records | 8.57M rows, 3.04M positives | **recall ceiling 0.7968, stage-1 F0.5 0.8482** | Blocking collapses at full density (US 0.742, India 0.879). Stopped. |

**Root cause (exp 7):** the absolute 3-gram df caps (4000 name / 2000 address) introduced in run 2 for speed are 10x more aggressive at full density than on S10: at 1.3M US S1 rows, every city/street 3-gram exceeds 2000 docs and is dropped. S10 validation hid this (its S1 index is 10x smaller). **Test has full density (US 663k, India 810k S1), so v1 on test was also hurt.** Lesson: validate blocking at the same S1 density as test.

### Blocking at full density (`src/blocking_exp.py`, 50k held-out US rows vs all 1.32M US S1)

| Search | Recall | Search time / 50k rows |
|---|---|---|
| name char3, df cap 4000, top 10 | 0.555 | 6 s |
| name char3, df cap 20000 | 0.758 | 96 s |
| address char3, df cap 2000 (v1/v2 setting) | 0.412 | 2 s |
| address char3, df cap 10000 / 40000 | 0.882 / 0.930 | 42 s / 276 s |
| **address word 1-2 grams**, df cap 5000 | **0.894** | **3 s** |
| name word 1-2 grams, df cap 5000 | 0.655 | 2 s |
| name+address char3, cap 10000 / 40000 | 0.965 / 0.988 | 66 s / 446 s |
| union: address word + name word (5000) | 0.968 | 5 s |
| name+address word 1-2 grams, cap 5000, top 10 / top 20 | 0.976 / 0.982 | 5 s |
| **chosen:** name char3 k10 + address word k10 + name+address word k20 | **US 0.9856**, **India 0.9593** (+ embedding search for non-Latin names, which the India number excludes) | ~14 s |

### Run 9 — FULL with the new blocking (25 Sep 08:52–12:55)
Build 145 min (42 chunks, ~31 candidates per S2/S3 row, 319M pairs). Stage-1 sample 11.5M rows / 3.76M positives.

| Stage | Recall ceiling | OOF macro F0.5 (1.1M eval S1, full density) |
|---|---|---|
| Stage 1 (t=0.925) | **0.9851** (exp 7: 0.7968) | 0.9724 |
| Stage 2 (t=0.575) | 0.9851 | 0.9788 (+0.0064; S1-side competition matters much more at full density than on S10, where it gave +0.0014) |
| Expected-F0.5 set selection (floor 0.3, alpha 1.0) | 0.9851 | **0.97904** — chosen |

Note: S10 scores (0.9877) were optimistic; FULL (0.9790) is the honest estimate for test density.

Error analysis of run 9 (`work/full_errors.log`), points lost: wrong_s1 0.0092 (65% no address), under_thresh 0.0077 (22% no address; noisy house numbers "11" vs "127 Lindsey Avenue"), blocking_miss 0.0051 (**51% no address**; name-only search at full density misses chains), decoy_merge 0.0049.

Next ideas, ranked by expected gain / cost:
1. Sibling-agreement stage-2 features (`train full sib`, reuses cached stage 1; ~20 min) — targets decoys + noisy house numbers.
2. Rows without address: name word search with k=100 (needs rebuild, ~5 h) — targets half of blocking misses; many are ambiguous chains, so realistic gain maybe +0.0005–0.001.
3. Stage-2 LightGBM tuning (more rounds, lower learning rate) on the cached base.

**Decision (exp 8):** replace address char-3-gram search with word 1-2-gram search; add name+address word search (top 20); keep name char3 (top 10, top 30 without address) and the embedding search. New features: `cos_addr_w`, `cos_comb_w` (+ margin, rank), `from_comb`. One US chunk at full density: search 77 s + features 104 s, ~31 candidates per row. Rebuilding FULL train and test with it (~6.5 h chain, `work/run_all.sh`).

## Submissions plan 25 Sep (each upload tests one thing; saved as submissions/vN + git tag vN)
| Ver | Changes only | OOF (FULL) | Question the leaderboard answers |
|---|---|---|---|
| v2 | two-stage LightGBM, word-level shortlist, expected-F0.5 | 0.97904 | **LB 0.9677** (OOF - 0.011). OOF does not track LB 1:1 |
| v3 | + sibling-agreement stage-2 features (GPU XGBoost stage 2) | 0.98015 (+0.0011) | **LB 0.9700 (+0.0023)**: gain on test is 2x OOF gain -> decoy/group handling is the lever |
| v4 | stage 1 as deeper GPU XGBoost (depth 10, 1000 rounds) | running | is the model capacity-limited? |
| v5 | France-only threshold shift on best model | - | is France over/under-matched? |
| v6 | average LightGBM + XGBoost scores | - | do two model types add accuracy? |

### LB gap analysis (25 Sep 17:40)
- Test has 5.5-5.8 S2/S3 rows per S1 vs 4.68 in train; predicted matches per S1 on test (3.37) ~ train truth (3.46) -> test has ~1.9x more unmatched look-alike rows per S1.
- `decoy_sim.py`: duplicating unmatched OOF rows to the test rate drops v2 OOF 0.9790 -> 0.9773 only; re-tuned rule gains +0.0002. So decoy rate explains ~0.002 of the 0.011 gap.
- Remaining ~0.009 unexplained: prime suspect France (15% of test, no labels): France F~0.92 with US/India ~0.977 would give 0.968.

### Test-side EDA (25 Sep 18:40) — what differs between train and test
- **Unsure band grows on test in every country**: best-candidate probability 0.05-0.5 = 3.0-3.8% of records in train OOF vs 7.8-8.9% on test (India 7.8, US 8.9, France 8.7). So the gap is not only France.
- The unsure test records are mostly **same name + same street + house number shifted by a few units** (9692 vs 9687 Diamond Rd; 657 vs 652 39th Ave; 441 vs 432 Inspiration Ln).
- In train, such pairs (name token-set >= 85, address token-set >= 70, first house number within 3%, not equal, not a prefix) are **92.7% false** (look-alikes); same-number pairs are 97.1% true.
- Share of records whose best candidate is such a shifted-number look-alike: US **14.3% train vs 21.9% test**, India 2.2% vs 3.2%, France 2.0%.
- Confident pairs (p1 >= 0.9) look the same in train and test (US name/address token-set 93.9/94.9 vs 93.9/94.5; India 77.6/95.3 both) -> test true matches are **not** noisier.
- France confident pairs: 91.8 / 91.0 -> France behaves like the other countries.
- **Density shift**: chain count (S1 rows sharing the core name) of best candidates: US train 22.0 vs test 11.7 (test US S1 is half as dense); India 19.4 vs 17.9. Features tied to density are outside the training range for US.

Implications: (1) the look-alike share is the main measurable difference; (2) US density differs; (3) France does not look worse than the others on any measurable statistic.

### Fable review checks (25 Sep 19:10)
- Check 3, ID leak: Spearman(S1 id number, matched id number) = 0.0003; sibling S2 ids are ~165M apart (median). No leak.
- Check 5, per-source cap: matches per (S1, source) = 1: 1.51M, 2: 1.32M, 3: 0.71M, 4: 0.26M, 5: 59.5k, 6: 2.8k. No useful cap.
- Check 6, calibration of v3 p2 (best candidate per record): under-confident in the middle (predicted 0.45 -> actual 0.56; 0.15 -> 0.23; 0.86 -> 0.92). Isotonic calibration before the expected-F0.5 rule is worth testing (calibrate on test-like OOF).
- Check 4 (test look-alikes = train businesses?): pending, needs RAM after v4 training.
- Forum rules (25 Sep): hand-written normalisation dictionaries, unsupervised stats on test, self-training allowed; libpostal/gazetteers/APIs not allowed. Plan in docs/AWS_PLAN.md.

### v5 result and France diagnosis (25 Sep 19:00-19:30)
- **v5 (v3 with every France S1 row emptied) LB 0.838.** Test S1: France 259,452 of 1,732,544 (15.0%). Train singleton share 5.6% in both countries.
  France F0.5 = 0.132 / 0.1497 + 0.056 = **~0.937** (0.934-0.941 given LB rounding); US+India = (0.970 - 0.1497 x 0.937) / 0.8503 = **~0.976**.
  So v3's OOF-LB gap (0.980 vs 0.970) is ~0.006 France + ~0.004 US/India (test density / look-alikes).
- France is not under- or over-matched: predicted matches per S1 France 3.43, US 3.38, India 3.34 (train truth 3.46); S1 non-empty 94.8% vs 94.3% (train 94.4%).
- France's best-candidate probability is not more uncertain: 0.05-0.5 band France 8.5%, US 8.8%, India 7.1%. Its loss is confident mistakes.
- **France S1 twins**: same generic name ("<city> <word> SARL"), same street, differing only in legal form (SA/SAS/SARL/SASU) or house number. Share of records whose 2nd candidate also has p1 > 0.5: France 2.07%, US 0.39%.
- France S2/S3 noise not handled by the US-trained cleaning (tokens in records but not in their confidently matched S1): dotted legal forms (s.a.s., s.a.r.l., e.u.r.l., s.a.s.u. ~5% of records), "et" for "&", street abbreviations r./av./ave/st./all./bd./blvd/imp./rte./crs/q./pl./ch./psg., "No"/"N°" before the number, departement names (Nord, Gironde, Loire-Atlantique, Pas-de-Calais) where S1 has region names.
- Decoy name words are already handled: US records with Midtown/Northside/Greater/Eastgate are 0% true in train and the model gives p2 > 0.5 to 0.1%; France records with Participations/Holding/Distribution/International get p2 > 0.5 for 0.9-2.4%.
- No Paris/Lyon/Marseille in France test (cities: Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, ...); 5-digit postcodes in 0.4% of S1 addresses.
- **Check 4 (are test records copies of train S1 businesses?)**: 60k US test records: best train-S1 cosine >= 0.9 while best test-S1 < 0.6 for 0.00%; test S1 with a train S1 at cosine >= 0.9: 0.09%. No overlap -> the "empty records that belong to train businesses" idea is dropped.
- Only India records have embedding features (non-Latin names); US and France both have none, so this is not France-specific.

### v4 = training on test-like conditions (25 Sep 18:59-19:25)
`testlike.py`: US S1 kept at test density (50%), India 91.7%; look-alike records topped up to test's records-per-S1 (US 5.76, India 5.10). Same features, GPU XGBoost both stages + sibling features.
- Held-out on the test-like setup: stage 1 0.9719, stage 2 0.9775, recall ceiling 0.9844; best rule plain threshold 0.55 (expected-F0.5 rule 0.9759 there). Not comparable with v3's 0.9802 (different validation data).
- France cleaning rules written (`normalize.py` french=True, `prep.py france` -> split `testfr`), not yet used.

## Lessons (read before changing anything)
- GPU 4 GB cannot fit XGBoost on the full 11.5M-row stage-1 sample (OOM after 3 folds, v4 first try). Use fold models (saved immediately) and average them for test.
- A Claude Code session restart kills background jobs. Every long step must be resumable: build skips chunk files already on disk (added 25 Sep 13:45 after the test build died at India chunk 6); stage 1 and test stage-1 scores are cached.
- A relative `max_df` makes search cost grow with S1 size; use an absolute document-frequency cap for **search**, but compute similarity **features** with the full vocabulary.
- Always time one chunk at test scale (US test S1 = 663k rows) before launching a full run.
- Prediction share per country on test (v1): India 12.8% empty vs 5.6% singletons in train, so India is under-matched.
