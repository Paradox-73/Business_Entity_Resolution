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

## Lessons (read before changing anything)
- A Claude Code session restart kills background jobs. Every long step must be resumable: build skips chunk files already on disk (added 25 Sep 13:45 after the test build died at India chunk 6); stage 1 and test stage-1 scores are cached.
- A relative `max_df` makes search cost grow with S1 size; use an absolute document-frequency cap for **search**, but compute similarity **features** with the full vocabulary.
- Always time one chunk at test scale (US test S1 = 663k rows) before launching a full run.
- Prediction share per country on test (v1): India 12.8% empty vs 5.6% singletons in train, so India is under-matched.
