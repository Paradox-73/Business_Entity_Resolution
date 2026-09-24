# AMLC 2026 — Strategy (read this first)

Updated 25 Sep 2026 ~00:45 IST, after measuring the real dataset.

## 0. Words used
- **S1 / S2 / S3**: the 3 sources. S1 = clean reference list. S2, S3 = messy copies from 2 vendors.
- **Match**: an S2/S3 row that is the same real business as an S1 row.
- **Singleton**: an S1 row with no match. Correct answer = empty list.
- **Candidate / shortlist**: S2/S3 rows we consider for an S1 row before the model decides.
- **Blocking**: the cheap step that builds the shortlist (comparing every pair is impossible at this size).
- **Recall ceiling**: share of true matches that are inside the shortlist.
- **Pair model**: gives each (S1, candidate) pair a probability of "same business".
- **CV (cross-validation)**: score on held-out training S1 rows; our own honest score.
- **OOF predictions**: pair probabilities from a model that never trained on that S1 row.
- **LightGBM**: fast tree model for tables of numbers; runs on CPU.

## 1. Where things are
- `student_resource/` — **the dataset + organisers' tools.**
  - `dataset/train/`, `dataset/test/` — the TSV files.
  - `utils/validate_submission.py` — format checker. Run before every upload.
  - `Documentation_template.md` — the methodology doc we must fill.
- `2026_official/` — problem statement, rules PDF, explainer video, the original zip.
- `past_prep_2025_price_task/` — old prep for a price-prediction task. Outdated; only the habits in §7 carry over.

## 2. What the data says (measured on the real files)

| Fact | Number | What it means for us |
|---|---|---|
| Train size | S1 2.21M, S2 5.03M, S3 5.29M rows | Big. Everything must run in chunks; train the pair model on a sample of S1 rows |
| Test size | S1 1.73M, S2 4.89M, S3 5.08M | Test prediction alone is a multi-hour job. Time it on Day 1 |
| Singletons | **5.6%** of S1 | Few free points. Most of the score is finding matches correctly |
| Matches per S1 | average 3.46; most have 2–6; S2 and S3 each give ~1–3 | Recall matters too, not just precision |
| Can one S2/S3 row match 2 S1 rows? | **Never** (0 of 7.6M) | Give each S2/S3 row to at most its single best S1 row. Strong precision rule |
| Unmatched S2/S3 rows | ~26% of S2/S3 | Decoys exist; "has a similar name" is not enough |
| Country same in matched pairs | **100%** | Only compare within the same country. Cuts work by ~2–3× |
| S1 rows sharing an exact name with another S1 row | **39%** (e.g. "Primary Care Group" ×253, "N & K Green" at 3 addresses) | Name alone can't decide. **Address (house number, street, city) decides** |
| Exact name equal in true pairs | only 22% | Need fuzzy similarity, not exact keys |
| Indian-script text | ~13% of India S2 names in Hindi script; ~13% of India S2/S3 addresses contain Hindi/Gujarati/Telugu (often the state name); S1 never | Need script-to-Latin handling for India |
| Test countries (S1) | India 810k, US 663k, **France 259k (15%)** | France has no training labels. Rules must be country-neutral |
| Missing address | ~3–5% of S2/S3 | Name-only pairs need their own handling |

Noise seen by eye in matched groups:
- Names: added words ("Mr", "Sri", "Center"), fake aliases ("Miradrex d/b/a Bharari Aviation…", "Nylacaloio a/k/a Zenith"), web-domain form ("daynightvardhman.com", ~3–4%), brackets ("[LLC]"), random accents ("Límited"), letter swaps ("lnflection", "Avatno"), words dropped or reordered.
- Addresses: S2 is all-caps; S3 uses state codes ("GJ", "MH") or full state names; parts reordered; parts missing; house number cut ("563" → "56"); junk tokens ("##", "NULL", "N/A"); state written in local script.
- Decoys named in the explainer video (`2026_official/problem_explainer_video_transcript.txt` [02:24]): **a similar name at a different address**, and **a different business at the same address**. So neither name nor address alone is enough; the pair model needs both.
- France (test only): "SARL/SAS/SCI" legal forms, "Rue/Avenue/Bd/R./Q." street words, accents, region names ("Nouvelle-Aquitaine"). The noise generator looks the same as for US/India.

## 3. How scoring works for us
- Score per S1 row: `F0.5 = 1.25·P·R / (0.25·P + R)`, then averaged over all S1 rows.
- Typical S1 row has 4 true matches:

| We predict | F0.5 |
|---|---|
| all 4 right | 1.00 |
| 3 right, 1 missed | **0.94** |
| 2 right, 2 missed | 0.83 |
| 4 right + 1 wrong | **0.83** |
| 4 right + 2 wrong | 0.71 |

- One wrong extra costs about 2× a missed one. **When unsure, leave it out.**
- Pick the cut-off by running this exact score on OOF predictions, never by accuracy.

## 4. The plan (5 stages)

**Stage A — Clean text (C).** One cleaning function used everywhere:
- Lowercase, strip accents, remove punctuation and junk tokens (NULL, N/A, ##).
- Split web domains ("bharariaviation.com" → "bharari aviation").
- Split aliases on "d/b/a", "a/k/a", keep both halves.
- Remove legal forms (Inc, LLC, Pvt Ltd, Limited, SARL, SAS…) into a separate field.
- Short forms learned **from training pairs**: which word in S2/S3 lines up with which word in S1 ("rd"↔"road", "gj"↔"gujarat"). Same for Indian-script state names (small, learnable list).
- Pull out parts: house number, postcode, city, state.
- Hindi-script names: transliterate to Latin (learn character mapping from training pairs, or a transliteration library if allowed — see form question 6).

**Stage B — Shortlist / blocking (A, with B).** Search **per S2/S3 row** for S1 rows, within the same country. Union of:
- Name character-trigram TF-IDF (splits text into 3-letter pieces, rare pieces count more), top ~10 nearest S1 rows.
- Address character-trigram TF-IDF, top ~10.
- Same city + same house number.
- Later (B): multilingual text embeddings (small model that turns text into numbers; handles Hindi and French) — `intfloat/multilingual-e5-small` (MIT).
- **Target: recall ceiling ≥ 97%** on train, shortlist ≤ ~15 per S2/S3 row. Report per country.

**Stage C — Pair model (C features, D trains).** LightGBM on each (S1, candidate) pair:
- Name scores: several fuzzy similarities (rapidfuzz library, MIT), TF-IDF similarity, rare-word overlap, alias-half match.
- Address scores: house number equal / conflicting / missing, street similarity, city equal, state equal, postcode.
- Context: rank of this S1 among the S2 row's candidates, gap to the best one, how many S1 rows share this exact name (chain-name risk).
- Source (S2 or S3), address missing flag.
- Train on a sample of ~300k S1 rows; 5 folds split by S1 row.

**Stage D — Decide (D).**
1. Each S2/S3 row goes to its **single best** S1 row, or to none if best probability < cut-off.
2. Collect per S1 row → output list.
3. Tune the cut-off (and separate S2/S3 cut-offs) on OOF with the real score.
4. Later: extra check that S2 and S3 rows given to the same S1 row also resemble each other.

**Stage E — Stronger model (B, Day 2).** Fine-tune a small multilingual cross-encoder (reads both records together and outputs a match probability) on hard pairs: `xlm-roberta-base` (MIT, 278M) or `mdeberta-v3-base` (MIT, 278M). Use its score only on the top 2–3 candidates per S2/S3 row (keeps GPU time low), as one more LightGBM input. Keep only if CV improves.

**France check (C + D, Day 2).** Train on US only → score on India, and the reverse. Any feature that breaks here is country-specific; fix or drop it. Also eyeball 50 French predictions.

**Not doing:** big 7–8B language models on every pair (too slow for ~10M rows on our GPUs); anything that looks things up online (disqualification).

## 5. Roles

| Person | Hardware | Owns |
|---|---|---|
| **A (captain)** | RTX 3050 4 GB, 16 GB RAM | Stage B shortlist, full pipeline, test-time run, **only uploader**, submission log |
| **B** | RTX 4060 8 GB + Kaggle | Embeddings for Stage B, Stage E cross-encoder |
| **C** | RTX 3050 Ti 4 GB + Kaggle | Stage A cleaning, Stage C features, France check |
| **D** | CPU + Kaggle | Scorer, folds, LightGBM, Stage D cut-offs, error analysis, methodology doc |

- All code in one folder shaped like the zip: `code/business_entity_resolution/src/`.
- Everyone uses the same S1 sample and the same folds file.
- Memory: 16 GB laptops can't hold all pairs at once. Process one country (or state) at a time. Kaggle gives ~30 GB RAM for bigger runs.

## 6. Timeline (round ends Sun 27 Sep 23:59 IST; 5 uploads/day)

| When | Goal |
|---|---|
| Fri morning | Everyone reads this. Scorer + folds + cleaning v1 (D, C) |
| Fri afternoon | Simple baseline: TF-IDF shortlist + "best S1 if similarity > x" rule. **Upload 1.** Time the full test run |
| Fri night | LightGBM v1 + exclusive assignment + tuned cut-off. **Upload 2** |
| Sat | Better cleaning (aliases, domains, Hindi script, learned short forms), embeddings in shortlist, France check. **Uploads 3–6** |
| Sat night | Cross-encoder training on Kaggle (B) |
| Sun 12:00 | **Freeze.** No new ideas |
| Sun 12:00–18:00 | Rerun whole pipeline from raw files with one command; confirm same output |
| Sun 21:00 | **Final upload** (chosen by CV, not public leaderboard) |
| Sun 22:30 | Zip uploaded: `output/` (both TSVs) + `code/` + filled `Documentation_template.md` |

- Save every uploaded file in `submissions/` with time, CV score, leaderboard score, note.

## 7. Past lessons that still apply
- Trust CV over the public leaderboard (2025: the public board was a third of test; final order differed).
- Log every experiment: change, CV score, difference. Becomes the doc's results table.
- Keep 1–2 real error examples with the fix (the 2025 winner's example was named by a judge).
- Small, explained models; know runtime and model sizes by heart.
- Only A uploads. Validate before every upload.

## 8. Google Form questions
1. Are AI coding assistants (e.g. Claude Code, Copilot) allowed during the round?
2. Which submission counts for the private leaderboard: last, best public, or one we pick?
3. When does the 5-per-day limit reset, and do uploads rejected by validation count?
4. Guidelines say a 1–2 page doc; the problem statement says no page limit. Which applies?
5. Does the "MIT/Apache-2.0, ≤ 8B" rule cover helper models (e.g. a text-embedding model used for blocking), or only the final matcher?
6. Can we use an offline, open-source transliteration library (Hindi script → Latin) and hand-written abbreviation lists (e.g. "St" → "Street", French "Bd" → "Boulevard"), or does that count as external data?
7. Can we use the unlabelled test source files for unsupervised steps (word frequencies, TF-IDF vocabulary, nearest-neighbour search)?
8. Is the public/private split random by S1 entity, and does the public part include France?
9. Is there a time or hardware limit for reproducing our pipeline from the zip?
10. Where and by when is the final zip uploaded — the same deadline as the round?

## 9. Needed from the team
- Team name exactly as registered (zip name).
- RAM size of B's and C's laptops.
- Sleep shifts for Fri and Sat night.
- Google Form answers, pasted here when they come.
