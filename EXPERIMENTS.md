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

### v6 ingredients (25 Sep 20:00-20:25), all on the test-like validation (v4 base 0.97753)
| Change | Held-out macro F0.5 | Decision |
|---|---|---|
| Calibration: isotonic map of p2 (cross-fitted by fold), then expected-F0.5 rule per country | 0.97757 -> 0.97811 (+0.0005); US 0.9793 -> 0.9797, India 0.9762 -> 0.9768 | keep (floor 0.4-0.5, alpha 1.0; France uses the pooled rule) |
| p2 calibration check | p2 0.3-0.4 true 47.8%, 0.4-0.5 57.9%, 0.5-0.6 68.2%, 0.6-0.7 77.3%, 0.8-0.9 92.2% | p2 is under-confident, as in v3 |
| Consensus stage-2 features (agreement with ALL rows pointing at the same S1 with p1 >= 0.5: mean/min name and address token-set, share/count with the same house number) | stage 2 at t=0.55: 0.9775 -> 0.9791; with expected-F0.5 rule 0.97926 (+0.0017) | keep |
| France cleaning (French rules, testfr rebuild), v4 model | no labels; 3.1% of France matched pairs change (18,050 records lose a match, 27,283 gain one, 816 move); best p2 > 0.9: 59.0% -> 60.1% | LB only |
| Self-training on test (1.03M pseudo-labelled test rows incl. 519k France, weight 0.5; validation adds each fold's own pseudo-labels) | 0.97921 vs 0.97926 without | **not used** (no gain on held-out; its test effect cannot be measured) |
| Calibration refit on the consensus model | 0.97915 (thr 0.55) -> 0.97966 (US 0.9809, India 0.9786) | **v6** = consensus model + calibrated per-country rule + France from the cleaned rebuild; 1,637,900 S1 rows with matches (US 2.33M, India 2.76M, France 0.91M matched records) |

Smoke tests for AWS (laptop GPU, tiny subsets): `embed_all.py` train/encode/search/eval and `rerank.py` select/train/score/stage3 run end to end.
Close calls in v4 scores: train 1.60M of 7.95M S2/S3 rows, test 2.90M of 9.97M (3.43M pairs).

### v4 LB 0.961357 (25 Sep 20:50) — test-like training HURT (-0.0086 vs v3)
- Cause (checked on data): `testlike.py` removed S1 rows by deleting their candidate pairs, so records kept only the
  remaining candidates. Candidates per US record: full train 31.4, **test-like 15.9**, test 31.3; candidates with name
  token-set >= 90: full 3.4, test-like 1.87, test 3.84 (India 30.1 / 27.5 / 30.1). The real search at lower density
  refills each record's top-k with other S1 rows, so test looks like FULL train on these features, not like test-like.
- The test-like held-out score (0.9775) could not see this: its validation rows have the same distortion.
- Decision: drop test-like training. v6a (built on it) not uploaded. v6 rebuilt on FULL data (v3 base) + consensus
  features + calibration + France cleaning.

### v6 on FULL data (25 Sep 20:50-21:40)
| Change (FULL validation, same as v3) | Held-out macro F0.5 | Decision |
|---|---|---|
| v3 (reference) | 0.98015 | LB 0.9700 |
| + consensus stage-2 features (LightGBM stage 1 of v3, XGBoost stage 2) | 0.98151 (+0.0014); `cons_num_share` is the 3rd most important stage-2 feature | **v6** |
| + calibration, per-country rule | 0.98173 (+0.0002 more) | left out: on test it adds ~56k borderline matches (p2 0.4-0.5), the band where test has 2x train's look-alike share |
| v6 = consensus + v3's rule (expected-F0.5, floor 0.3, raw p2) + France from the cleaned rebuild | - | 1,635,015 S1 rows with matches; matched records vs v3: US 2.261M (+20k), India 2.746M (+39k), France 0.908M (+19k) |

### Held-out loss breakdown of the best model (full_cons, 0.98151; `error2.py`, log work/errors_full_cons.log, 25 Sep 23:30)
| Mistake | Points | Where |
|---|---|---|
| true record never reaches stage 2 | 0.00694 | no-address 0.00422; chain names (6+ identical S1 names) 0.00404; shortlist recall 98.51%, stage-2 input 97.86% (0.65% cut by top-2) |
| best candidate right, rejected by the rule | 0.00692 | changed house number 0.00290; no house number 0.00196 |
| look-alike merged | 0.00262 | same house number, changed name 0.00125 |
| wrong S1 chosen | 0.00128 | no-address 0.00104 |
| record of another S1 merged | 0.00073 | |
- No-address records cost ~0.0073 of 0.0185; most are chain names with no way to pick among identical S1 rows (a floor for everyone).
- House-number change patterns are already learned: p2 matches the true rate per bucket (diff 1-2: p2 0.193 vs true 0.211; 3-5: 0.053 vs 0.050; >50: 0.564 vs 0.576). New number features would not add information.
- France legal-form veto (audit, confirmed): France pairs with legal forms on both sides conflict in 3.6% of p2>=0.3 best candidates (US true pairs: 0.8%); 1,034 of 1,124 same-name same-number conflicts belong to S1 rows that already have a record with the matching form. v7 = v3 + this veto (17,934 pairs set to 0; France matched records 889,209 -> 883,826).

### Shortcut checks (25 Sep 23:40-23:55) — all negative
- ID numbers: |S1 id - record id| of true pairs has the same distribution as random pairs (share < 10M: 1.99% vs 1.99%; median 293M vs 293M). No ID leak.
- File row order: Spearman(S1 row, record row) = -0.002 / -0.001; records of the same business are ~1.1M rows apart. No order leak.
- Per-source coverage: 85% of matched S1 have records from both sources; the model already has same-source counts. Weak.
- No-address records whose true S1 shares its core name with 2-60 S1 rows (random pick right 26.2%): closest ID 26.4%, most other records 23.1%, closest raw name (legal form, punctuation kept) 42.3%. Only the raw name carries signal, too weak to pay under F0.5.
- No-address share: train US 3.6% / India 3.0%; test US 2.9% / India 2.4% / France 3.0%.

### Night 25->26 Sep results
| Run | Held-out macro F0.5 (FULL eval S1) | Decision |
|---|---|---|
| Transformer reranker (multilingual-e5-small cross-encoder, word table frozen, 827k close-call rows of the non-eval half, 1 epoch) + stage-3 XGBoost on v6 (full_cons) p2 | GBDT 0.98094-0.98114 -> **0.98794** at threshold 0.65 (+0.0068); stage-3 gain: transformer score 6.0M vs p2 1.2M | **v7** = this + France legal-form veto (1,633,377 S1 rows with matches) |
| Stage 2 trained on all Latin-name labels (both halves), rate 1.0, sibling features | 0.98089 vs v3 0.98015 (+0.0007) | candidate for combining with the reranker |

### Test-like split tl2 (train S1 thinned to test's count BEFORE the search; US/India only; out-of-fold; 26 Sep 03:40)
| Model | tl2 macro F0.5 (all / India / US) | FULL held-out | LB |
|---|---|---|---|
| v2 (`full`) | 0.97540 / 0.97506 / 0.97581 | 0.9790 | 0.9677 |
| v3 (`full_sib`) | 0.97651 / 0.97681 / 0.97615 | 0.98015 | 0.9700 |
| v6 model (`full_cons`) | **0.97798** / 0.97858 / 0.97725 | 0.98151 | 0.9673 (with France cleaning) |
- tl2 sits ~0.004 below FULL, close to the US/India LB level (~0.976 from the v5 probe), and ranks v2 < v3 like the LB.
- v6's model is +0.0015 over v3's on tl2, so v6's LB drop most likely came from its France cleaning, not the consensus features. v7 keeps v6's model for US/India and the old (v3) France cleaning.

### Transformer versions, night 26 Sep (rerank.py; held-out = eval-half S1, best of 17 decision rules)
| Version | Base model | Transformers | Held-out: GBDT -> with transformer | Notes |
|---|---|---|---|---|
| v7 | v6 (`full_cons`) | A (trained on non-eval half) | 0.98094 (thr 0.65) -> 0.98794 (thr 0.65) | built 02:51 |
| **v7b** | v6 | A + B (B trained on the eval half; stage 3 learns from both halves; test = mean of A's and B's stage-3 probabilities) | 0.98151 -> **0.98805** (expF floor 0.5); all S1 0.98157 -> 0.98808 | A and B disagree at 0.5 on 44k of 2.76M test pairs; the averaging is not measurable on held-out |
| v7b_frbase | v6 | as v7b, France rows without transformer | same | France matched records 893k vs v7b 877k |
| **v7c** | v6 | as v7b; France: p = min(GBDT p2, stage-3 p) | same | France 861k matched; see audit below |
| v7sib | v3 (`full_sib`) | A + B | 0.98015 -> 0.98780 | v3 base narrows the gap to v6 base from 0.0014 to 0.0003 |

| v7blend | 0.7 v6 + 0.3 v3 reranked | A + B | 0.98801 (weights 0.5/0.5: 0.98795) | no gain over v7b; not recommended |
| v7d | v6 | A + B, close calls + stage-1 ranks 3-5 (`BER_S3_EXTRA=1`) | 0.98151 -> 0.98811 (all S1 0.98815) | +0.00006 over v7b/v7c; changes 0.4% of test S1 rows vs v7c; France min rule |
| (stage 3 + segment features) | v6 | A + B | 0.98806 | no gain; not built |
| (A2 alone) | v6 | e5-base (A2), 420k rows of the non-eval half, 16-bit frozen word table, batch 16 | 0.98790 | about equal to e5-small A (0.98794) with half the data |
| **v7ens** | v6 | 0.7 x (A+B stage 3) + 0.3 x (A2 stage 3); France min rule | **0.98813** (0.5/0.5 also 0.98813) | best held-out; changes 0.4-1.5% of test S1 rows vs v7c (France 3,797, India 3,099, US 2,605). **LB 0.983159** (26 Sep; v3 0.9700; leader 0.990621) |

**France checks, 26 Sep 15:40-17:00 (probe v7ens_frempty LB 0.848792 -> US/India ~0.9887, France ~0.952; France = ~70% of the gap):**
| France hypothesis (no France labels) | Label check on US/India train | Verdict |
|---|---|---|
| Descriptor swap ("Calais Amicale" -> "Calais Services") marks fakes (v7h) | swap candidates are TRUE 98.9% (US) / 98.2% (India): drop a word + add a generic suffix is normal noise | **refuted, v7h not uploaded** |
| Legal-form conflict marks fakes (veto in v7a..v7ens) | conflicts TRUE 5.2% (US) / 0.0% (India) | **confirmed, kept** |
| France rejects noisy true records (acceptance per change pattern far below US true rates: swap+same number 66.8% vs 91.7%, same name+other number 13.9% vs 42.7%) | per-pattern recalibration costs 0.002 on labelled US (0.98691 -> 0.98493) and would accept 1.01M France records (half strength 0.95M) vs ~0.88M expected true (3.4 per S1 as in US/India) | **refuted by the count bound, not built as a version** |
| Transformer additions in France are fakes (min rule in v7c..v7ens) | its basis compared added pairs with "kept by both" (easy pairs), not with true pairs; US true pairs change house number 10.8% | **basis flawed -> v7i removes the rule (+14.2k France records, 874.9k vs ~0.88M expected)** |
- France's accepted count (861k in v7ens) is already ~97.6% of the expected true count, so France loses by picking partly the wrong records (~4% wrong, ~6% missed), not by a threshold. The French address cleaning of v6 looks correct on samples but its LB went down; not reused.
- v7i (transformer may add France matches) LB 0.982636: **France -0.0035** -> additions are net wrong; min rule stays.
- Same-name S1 with the record's exact address while another S1 was chosen: 3 cases in US train, 0-5 in France test -> not a France error source.
- **Live hypothesis (19:20):** France rejects number-changed true records. v7ens France acceptance of best candidates: same name + other number 20.6%, word swapped + other number 1.0%, word added + other number 0.1%; US true rates 42.7% / 16.1% / 6.0%. `france_cal.py ndiff` adds (highest probability first, never vetoed pairs) halfway to the US rates: **v7j** +21,155 France records = 881.9k (= expected ~882k); v7j_full +42,309. Only France differs from v7ens, so LB change / 0.15 = France change.

**26 Sep 15:00, next design (3-fold transformers):** fold side k trains on all close calls whose record is in another pipeline fold (incl. "mixed" records and stage-1 ranks 3-5): 1.68M pairs per model vs 0.83M per half-model, and every close call gets an out-of-fold score. Laptop: e5-small (`ce_folds.py --name small`); RTX 4060: e5-base (`docs/FRIEND_RUN.md`). Stage 3 prints the "halves protocol" score (mixed records keep GBDT p2), comparable with v7ens 0.98813.
- **Early check 16:30** (`work/cmp_fold0.py`): e5-small fold-0 model vs transformer A (v7ens) on the SAME 280,159 held-out pairs (eval group, fold 0, neither trained on them): AUC 0.98865 -> **0.99113**, log loss 0.13020 -> **0.11716**, best candidate is the true S1 0.98568 -> **0.98657**. Same model size, so the gain is the 3-fold design (2x data, mixed records, ranks 3-5). Green light for the e5-base run.

**Transformer at test-like density (tl2, `rerank.py check`, side A only, stage 3 fitted on FULL rows of other folds; 26 Sep 05:40):**
| Base | tl2 GBDT | tl2 + transformer | US | India |
|---|---|---|---|---|
| v6 (`full_cons`) | 0.97798 | **0.98747** (+0.0095) | 0.97729 -> 0.98760 | 0.97858 -> 0.98739 |
| v3 (`full_sib`) | 0.97651 | 0.98724 (+0.0107) | 0.97615 -> 0.98733 | 0.97681 -> 0.98720 |
- The transformer gains MORE at test density than on FULL (+0.0065), i.e. it handles the extra look-alikes. tl2 tracked the LB for v2/v3 (US/India ~0.976), so US/India LB for v7b/v7c ~0.987 is the expectation.
- Rejected: France house-number veto. First-number parsing picks apartment/floor numbers ("Appartement 22", "etage 4"); 1.38% of v7c France pairs differ, many plausibly true. Not validated -> not used.

France label-free audit (v7b vs v7b_frbase, 26 Sep 05:20): the transformer changes 42,924 of 259,452 France S1 rows (16.5%; US 0%, by construction). Records it removes: house number differs from S1 20%, name word differs 73% (the known France look-alike pattern). Records it adds: house number differs 34%. Records both keep: 1.6%. So its France removals look right and its additions mostly wrong -> v7c keeps only the removals. The same profile on US/India is not informative (true US/India pairs change house number 16% of the time).

### Remaining held-out loss AFTER the transformer (v7b's p, eval S1, `error2.py full_cons_ce full`; 26 Sep 06:00)
| Mistake | Points | Count |
|---|---|---|
| true record never in stage-1 top 2 | 0.00681 | 81,531 (no address 50,893 = 0.00416) |
| right S1 found, rejected by the rule | 0.00258 | 31,933 |
| wrong S1 chosen | 0.00113 | 13,842 |
| record with no true S1 merged | 0.00083 | 2,999 |
| record of another S1 merged | 0.00060 | 2,285 |
- The transformer removed most "wrong S1" and look-alike loss; "never in top 2" is now 57% of what is left.
- Stage-1 ranks 3-5 of the reranker's records (`topk5.py`, p1 >= 0.005): train 441,400 candidates of 175,565 records with 13,569 true pairs; test 942,458 candidates of 384,621 records. Queued as v7d (stage 3 with these, `BER_S3_EXTRA=1`).
- e5-base transformer (A2) fits the 4 GB GPU with a 16-bit frozen word table and batch 16 (4.6-6.5 steps/s); queued as v7ens.

### Remaining held-out misses of full_cons (eval S1, GBDT expF rule; 26 Sep 03:20)
- Missed true pairs 169,650 of 3,815,794: in the reranker's close calls 86,694; never in stage-1 top 2 81,531 (reranker cannot see them); outside the close-call band only 1,425. Wrong predicted pairs 12,473: close calls 11,918, "certain" (p2 > 0.995) 555.
- Widening the close-call band would gain almost nothing. The next unreachable block needs stage 1 to keep more than 2 candidates per record.

### Blocking and cleaning audit (26 Sep 17:20-19:30, laptop; blocking only, no model retrained)
Setup for every number below: 100k matched train records per country with a Latin-script name whose true S1 is in the eval half (crc32 % 1000 < 500, never seen by the fine-tuned e5), searched against ALL train S1 rows of the country (full density). Recall = share of true pairs in the shortlist; the production numbers reproduce `candidates.py` exactly (US 0.9868, India 0.9784). India non-Latin names are excluded (they already have the e5 search). Scripts: `dense_blocking_exp.py`, `blocking_variants.py`, `name_search_diag.py`, `name_cap_exp.py`, `ordinal_exp.py`, `combined_blocking_exp.py`; logs in `work/`.

**All fixes together, added one at a time (`combined_blocking_exp.py`):**
| Step | US recall | India recall | Candidates/record (US / India) |
|---|---|---|---|
| production shortlist | 0.9868 | 0.9784 | 31.4 / 30.3 |
| + name search max_df 4000 -> 20000 | 0.9910 | 0.9836 | 31.5 / 30.4 |
| + word searches min_df 2 -> 1 | 0.9921 | 0.9852 | 31.4 / 30.2 |
| + no-address name search k 30 -> 100 | **0.9943** | **0.9884** | 34.5 / 33.3 |
| + dense e5 top-10 (fine-tuned `e5_ft_addr`, all records) | **0.9971** | **0.9956** | 42.0 / 41.4 |
| (production + dense e5 top-10 only) | 0.9941 | 0.9916 | 39.3 / 38.6 |
- Misses fall from 1.32% to 0.29% (US) and 2.16% to 0.44% (India). The three lexical fixes alone match dense-only with fewer candidates, no GPU, no labels.
- Not rebuilt, so the F0.5 effect is unmeasured. Estimate: the shortlist misses cost ~0.0047 held-out today (share of the 0.0068 "never in top 2" block); many recovered pairs are chain names with vague addresses, so expect +0.001-0.002 on US/India, not the full amount.

**Name search is starved at full density (`name_search_diag.py`, US, 30k records whose cleaned name EQUALS their true S1's and is unique in S1):**
| Name vectoriser | 3-grams kept per name (median) | names with 0 kept | top-10 recall on these exact-name records |
|---|---|---|---|
| min_df 2, max_df 4000 (production) | 4 | 3.2% | **0.860** (web domains 0.840) |
| min_df 2, max_df 20000 | 12 | 0% | 0.9998 |
- Examples with nothing left: 'thomastechnologies', 'safereal', 'superiordirect' (every 3-gram is in > 4000 US S1 names). Same failure as exp 7 for addresses (cap tuned on S10); the name search was never re-tuned. min_df makes no difference here.
- Cost vs gain (`name_cap_exp.py`, union recall, same candidate count): max_df 10000 US +0.0033 / India +0.0032, name search 43 s / 25 s per 100k queries; 20000 US +0.0042 / India +0.0052, 124 s / 65 s (production 6 s). France S1 is 5x smaller than US train S1, so its cap bites less.

**Dense e5 search for Latin names (`dense_blocking_exp.py`):**
| | US | India |
|---|---|---|
| dense top-10 alone | 0.9843 | 0.9727 |
| dense top-20 alone | 0.9876 (beats the whole TF-IDF union with 20 vs 31 candidates) | 0.9780 |
| TF-IDF misses recovered by dense top-10 | 728 of 1,320 (chain names 63%, no address 34%) | 1,316 of 2,161 (chain 76%, no address 22%) |
- Recovered pairs read as genuine (`work/dense_recovered_{US,India}.tsv`): squashed web domains ('novatrading.com' -> 'Nova Trading'), heavy typos ('Raj Ceonsultfnts'), spelled-out ordinals, word-order shuffles.
- Exact GPU search (RTX 3050 Ti): ~1 ms/query against 1.32M US S1; encoding 3,300-3,600 texts/s. Full train+test estimate ~5 h on the laptop GPU. FAISS IVFFlat is worse than exact at useful speeds (US recall@20 0.873 at nprobe 8, 0.920 at nprobe 32, vs 0.9876 exact); not needed.
- Off-the-shelf multilingual-e5-small on a 60k-S1 smoke run: about equal to the fine-tuned model for Latin names (not run at full size).

**Other blocking variants (`blocking_variants.py`, one change at a time vs production):**
| Variant | US union | India union | Cand/record change |
|---|---|---|---|
| word searches min_df 1 (hashing TF-IDF; min_df 2 hashing reproduces production 0.9870 / 0.9790) | +0.0017 | +0.0018 | 0 |
| BM25 instead of TF-IDF (word searches) | +0.0000 | +0.0009 | 0 |
| BM25 + min_df 1 | +0.0014 | +0.0030 | 0 |
| comb k 20 -> 30 / 40 | +0.0018 / +0.0029 | +0.0029 / +0.0048 | +9 / +19 |
| addr k 10 -> 20 | +0.0004 | +0.0001 | +6 / +4 |
| no-address name k 30 -> 100 | +0.0016 | +0.0020 | +3 |
| S1 initials added to the comb documents | +0.0000 (acronym records 0.965 -> 0.988) | **-0.0010** | 0 |
| no learned abbreviation map (does the map help?) | **-0.0019** | -0.0002 | +0.6 |

**Cleaning bugs found:**
- **Indian state codes.** `maps.json` was learned on US+India pairs pooled, so `tn` -> 'tennessee' (88k India train records end in 'TN'); `dl` (170k) and `od` (16k) are not mapped; India S1 writes 'Tamil Nadu' / 'Delhi' / 'Orissa'. ~274k India records (6.6%) carry a systematic address mismatch. Fix (tn -> tamil nadu, dl -> delhi, od/odisha -> orissa): blocking +0.0003 only (state names are above the df cap), but address token-set of affected true pairs (6,870 in the sample) mean 91.2 -> 97.5, share >= 90 0.698 -> 0.911. A feature fix, not a blocking fix. The same pooled map turns French 'de' / 'la' into 'delaware' / 'louisiana' (both sides alike, so mostly harmless).
- **Ordinals (`ordinal_exp.py`).** US records spell ordinals out ('eleventh street') in 1.3% of addresses vs 0.36% of S1, which writes '11th'. Normalising both sides (words -> digits, '11th' -> '11'): affected records (1.45%) recall 0.9352 -> 0.9834, their true-pair address token-set mean 89.7 -> 96.3 (>= 90: 0.664 -> 0.884); US union +0.0014.
- **France acronym records.** Short France names are often initials of the S1 name ('SDS' = 'Securite Darts Sport'): 53.6% of 29,729 short-name France records have an S1 with equal initials, the same first house number and address token-set >= 80, vs 16.1% with names shuffled (chance) -> ~11k acronym records (~0.8% of France records). Train has the pattern too (45% of matched short-name India records are initials of their true S1; US 7%). No current feature links them; worth at most ~0.0005 LB.
- Checked and fine: no junk-only addresses flagged as present (0 in every file); entity ids have no leading zeros, so the int round trip in `finalize.py` is exact; S1 halves and folds are consistent across `embed.py`, `pipeline.py` and `rerank.py`; the official validator is unmodified.

**Decision:** none of this reaches the leaderboard without a full rebuild of train and test pairs (~8 h) plus retraining stages 1-2 and re-scoring the new close calls with the transformers. If rebuilt, bundle in one rebuild: name max_df 10000-20000, word min_df 1, no-address k 100, state-code and ordinal cleaning, and dense e5 top-10 for all records (`embed.py encode_all` for every record, not only non-Latin). Otherwise these are the measured blocking alternatives for the methodology document.

### 3-fold transformers: e5-small (laptop) + e5-base (friend 1, RTX 4060, 800k rows/fold); 26 Sep 21:00-23:30
Same held-out pairs (labelled close calls whose record is in fold k; no compared model trained on them). AUC / top-1 = best candidate is the true S1 (records that have one):

| fold | old A+B (v7ens) | e5-small (1.68M rows) | e5-base (800k rows) | mean of small+base logits |
|---|---|---|---|---|
| 0 (557k pairs) | 0.98875 / 0.98586 | 0.99121 / 0.98670 | 0.99007 / 0.98687 | **0.99168 / 0.98698** |
| 1 (551k) | 0.98891 / 0.98576 | 0.99144 / 0.98653 | 0.99081 / 0.98655 | **0.99206 / 0.98679** |
| 2 (551k) | 0.98830 / 0.98568 | 0.99084 / 0.98676 | 0.98994 / 0.98639 | **0.99141 / 0.98687** |

Final held-out F0.5, halves protocol (comparable with v7ens 0.98813; `build_combo.sh`):

| version | what | halves | all close calls rescored |
|---|---|---|---|
| v7f | e5-small folds, stage 3 | 0.98838 (+0.00025) | 0.98903 |
| (family base) | e5-base folds, stage 3 | 0.98826 | 0.98890 |
| v7f2 | e5-small + old A+B + A2, equal blend | 0.98835 (old transformers add nothing) | - |
| **v7g** | stage 3 per family, blend 0.7 small / 0.3 base | **0.98845 (+0.00032)** | 0.98911 |
| v7g2 | mean logits of small+base (`avg_family.py small base sb`), one stage 3 | 0.98844 | 0.98911 |

- Each file comes in two France variants: `out_<v>` (v7ens France: transformer may only lower) and `out_<v>_num` (+ house-number veto, v7k rule): v7g France 852,055 -> 844,350 matched.
- **v7g_num LB 0.983103** (-0.000056 vs v7ens 0.983159). The file changed France on 16,774 of 259,452 France S1 rows (12,477 by the new transformers' France lowering alone: min(stage 3, GBDT p2) with a different stage 3 removes 10,789 and adds 2,101 France matches vs v7ens; the rest by the number veto) and US/India on 14,235 rows (0.96%). Mistake: 'v7ens France setting' meant the same rule, not the same France rows; two unvalidated France changes were uploaded together with the US/India change, so the loss cannot be attributed. Split: v7l = v7g US/India + v7ens France rows exactly (`splice_country.py`); v7l_k = v7g US/India + v7k France rows.
- **Uploaded 26 Sep ~23:30: v7g_num** (last upload of the day). LB - 0.983159 = 0.85 x US/India change (held-out +0.00032) + 0.15 x France change (new transformer's France lowering + number veto). Clean France split: upload `out_v7g` (same file without the veto).
- Friend 1's e5-base (800k rows) ranks below e5-small (all rows) on every fold: capping rows costs more than the bigger model gains. Friend 2 (A6000) trains bge-reranker-v2-m3 on all 1.68M rows, ~2.1 steps/s while another job (`xenc.py`) shared the card, 4.3 steps/s alone.

## Lessons (read before changing anything)
- **Windows power throttling slowed background jobs 3-5x (found 26 Sep 03:55).** Windows 11 runs windowless background processes on the slow efficiency cores of the i5-12450H. Transformer training: 2.3 steps/s throttled (GPU 34% busy) vs 11.6 steps/s after opting out (GPU 86%). `common.py` now opts every pipeline process out at start; `work/unthrottle.py <pid>` does it for a running process.
- **Resampled training data must be checked against test on every feature that depends on the candidate list** (candidates per record, counts, ranks, margins) before training on it. Deleting pairs is not the same as searching at lower density (v4: -0.0086 LB).
- GPU 4 GB cannot fit XGBoost on the full 11.5M-row stage-1 sample (OOM after 3 folds, v4 first try). Use fold models (saved immediately) and average them for test.
- A Claude Code session restart kills background jobs. Every long step must be resumable: build skips chunk files already on disk (added 25 Sep 13:45 after the test build died at India chunk 6); stage 1 and test stage-1 scores are cached.
- A relative `max_df` makes search cost grow with S1 size; use an absolute document-frequency cap for **search**, but compute similarity **features** with the full vocabulary.
- **Every search's df cap must be checked at full density, not only the one that failed.** The address cap was fixed on 25 Sep, but the name cap (4000, tuned on S10) left a median US name with 4 3-grams and 3.2% with none; exact unique names were found only 86% of the time (audit 26 Sep).
- Learned cleaning maps must be learned (or at least checked) per country: the pooled map sends India 'TN' to 'tennessee'.
- Always time one chunk at test scale (US test S1 = 663k rows) before launching a full run.
- Prediction share per country on test (v1): India 12.8% empty vs 5.6% singletons in train, so India is under-matched.
