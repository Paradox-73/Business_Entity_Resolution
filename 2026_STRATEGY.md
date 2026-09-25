# AMLC 2026 — ML workflow and strategy

Updated 25 Sep 2026, 17:50 IST. Detailed numbers for every run: `EXPERIMENTS.md`. Every uploaded file: `submissions/vN/` + git tag `vN`.

## 0. Words used
- **S1** = clean reference list. **S2/S3** = messy vendor records. **Match** = an S2/S3 record that is the same business as an S1 row.
- **Shortlist (blocking)** = the few S1 rows each S2/S3 record is compared with. **Recall ceiling** = share of true matches that make the shortlist.
- **OOF (held-out) score** = macro F0.5 on training S1 rows the model never trained on. **LB** = public leaderboard.
- **Look-alike (decoy)** = an S2/S3 record that matches nothing but resembles an S1 row.

## 1. Where we stand

| Version | What it tested | OOF | LB |
|---|---|---|---|
| v2 | two-stage model, fixed shortlist | 0.9790 | **0.9677** |
| v3 | + sibling features (compare a record with the other records already matched to the same S1) | 0.9802 | **0.9700** |
| LB leader | — | — | 0.9841 |

- Gap to leader: 0.014. Gap between our OOF and LB: ~0.010.
- v3's LB gain (+0.0023) is 2x its OOF gain -> anything that handles look-alikes pays off more on test than in validation.

## 2. The workflow, step by step (what was done, what it showed)

### 2.1 Understand the data (done)
- 2.2M S1 / 10.3M S2+S3 train; 1.7M S1 / 10M S2+S3 test. Test adds France (15%, no labels).
- Each S2/S3 record matches at most one S1; matched records always share the country; 39% of S1 names repeat (chains).
- Name differences: legal words (Pvt/Ltd) 25%, typos 13%, other script 7%, web domains 4-8%, invented names ~4%.
- **Test differs from train**: 5.7 S2/S3 records per S1 vs 4.7 -> ~1.9x more look-alikes per S1.

### 2.2 Clean the text (done)
- Accents, web domains, "d/b/a" aliases, legal words, junk tokens; address short forms and Indian-script state names **learned from training pairs**.

### 2.3 Shortlist (done; biggest single fix)
- Search from each S2/S3 record to S1 rows of the same country, union of 4 searches: name 3-letter pieces, address words, name+address words, and a **fine-tuned multilingual embedding** for Indian-script names (recall@10 0.16 -> 0.998, GPU).
- Recall ceiling at full size: **0.985**. Mistake found and fixed on the way: a 10% validation slice hid a recall collapse (0.797) at full size.

### 2.4 Pair model (done)
- 57 features per pair (name/address similarities, house numbers, how far a candidate leads the next one, embedding similarity).
- Stage 1: gradient-boosted trees on every shortlisted pair. Stage 2: re-scores each record's top 2 using S1-side competition + sibling agreement.
- Validation: 3 folds grouped by business; scored on the half of S1 not used to train the embedding model.

### 2.5 Decision (done)
- Each S2/S3 record goes to its single best S1; per S1, keep the set that maximises expected F0.5.

### 2.6 Error analysis (done at each step)
- Remaining OOF loss 0.020: wrong S1 chosen 0.009 (65% have **no address**), right S1 below cut-off 0.008, missed by shortlist 0.005 (half no address), false merge 0.005.

### 2.7 Explaining the OOF->LB gap (in progress)
- Simulating test's look-alike rate on OOF explains only ~0.002 of the 0.010 gap.
- **Main suspect: France.** If French rows score ~0.92 and US/India ~0.977, total is ~0.968 = our LB.

## 3. Next submissions (each answers one question)

| Ver | ETA | Changes only | Question | If yes -> | If no -> |
|---|---|---|---|---|---|
| v4 | ~19:00 | stage 1 = deeper GPU XGBoost | Is the model capacity-limited? | bigger models / ensemble | work on data, not model |
| v5 | right after | v3 with **France predictions emptied** | How much of the gap is France? (LB drop = France share x France score) | fix France: learn French address/name patterns from test structure | gap is US/India density shift -> retrain at test density |
| v6 | ~20:30 | best of v3/v4 + look-alike-aware training (more unmatched records per S1 in training) | Does training on test-like look-alike rates close the gap? | keep | revert |

## 4. Saturday (to reach 0.99)
1. Rows without address (half of shortlist misses): name-only search with a much wider list.
2. Look-alike handling: training data with test-like look-alike density; stronger sibling features.
3. GPU re-ranker (small multilingual model reading both records) for close calls only.
4. Final: ensemble of the best models, decision tuned under test-like conditions.

## 5. Rules we follow
- Every change: smoke test on 300k rows first, then full run; GPU for training/scoring.
- Every upload: format check (`check_submission.py`), saved in `submissions/vN/`, git tag `vN`, result logged in `EXPERIMENTS.md` + `submissions/LOG.md`.
- One change per upload, so each LB score answers one question.
