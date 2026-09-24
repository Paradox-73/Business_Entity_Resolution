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

## Lessons (read before changing anything)
- A relative `max_df` makes search cost grow with S1 size; use an absolute document-frequency cap for **search**, but compute similarity **features** with the full vocabulary.
- Always time one chunk at test scale (US test S1 = 663k rows) before launching a full run.
- Prediction share per country on test (v1): India 12.8% empty vs 5.6% singletons in train, so India is under-matched.
