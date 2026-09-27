# AMLC 2026 — Business Entity Resolution: full record of what was done

Updated 26 Sep 2026, ~15:00 IST (status table, §8 plan; §3–7 describe the day-1 pipeline, EXPERIMENTS.md has the transformer work). Written for cross-checking: every step, setting, number and known risk.
Sources for numbers: `EXPERIMENTS.md` (Appendix E), `submissions/LOG.md` (Appendix F), logs in `work/`.
The official problem statement, rules and video transcript are copied **word for word** in Appendices A–D.

## 0. Words used
- **S1** = clean reference list (one row per business). **S2 / S3** = two vendor sources with noisy copies. A **match** = an S2/S3 record describing the same business as an S1 row.
- **Singleton** = S1 row with no match (correct output: empty list).
- **Shortlist / blocking** = the few S1 rows each S2/S3 record is compared with. **Recall ceiling** = share of true matches inside the shortlist.
- **OOF (out-of-fold, held-out) score** = macro F0.5 on training S1 rows whose matches the model never trained on.
- **LB** = public leaderboard (part of the test set).
- **Look-alike / decoy** = S2/S3 record matching no S1 row but resembling one.
- **Stage 1 / stage 2** = first model scores every shortlisted pair; second model re-scores each record's top 2 using information about competing records.

---

## 1. Status

| Version | What it contains | Held-out score (validation) | LB | Uploaded |
|---|---|---|---|---|
| v1 | first pipeline (broken shortlist at full size) | 0.9753 (10% slice, optimistic) | — | no (team decision) |
| v2 | fixed shortlist + two-stage LightGBM + expected-F0.5 decision | 0.97904 (FULL) | **0.9677** | yes |
| v3 | v2 + 6 "sibling" features in stage 2 (stage 2 on GPU XGBoost) | 0.98015 (FULL) | **0.9700** (best so far) | yes |
| v4a | stage 1 as deeper GPU XGBoost | 0.98005 (FULL) | — | no (no gain on held-out) |
| v5 | v3 with every France row emptied (measurement probe) | — | **0.838** → France ≈ 0.937, US+India ≈ 0.976 | yes |
| v4 | trained on "test-like" data made by deleting pairs of dropped S1 rows (see §5.6) | 0.9775 (test-like, not comparable) | **0.9614** (−0.0086) | yes |
| v6a | v4 + consensus features + calibration + France cleaning | 0.97966 (test-like) | — | no (built on v4's broken base) |
| v6 | v3 base (FULL data) + consensus stage-2 features + France rows from French-cleaned rebuild | 0.98151 (FULL) | **0.9673** (−0.0027) | yes |
| v7probe | v6 US/India rows + v3 France rows | 0.98151 (FULL) | — | no (replaced) |
| v7a | v3 + France legal-form veto | 0.98015 (FULL) | — | not yet |
| v7 | v6 model + **transformer reranker** (e5-small cross-encoder, one half) on close calls + France veto | 0.98794 | — | not yet |
| v7b | v6 model + two cross-fitted transformers (A, B), expected-F0.5 rule + veto | **0.98805**; test-like split (tl2) 0.97798 → **0.98747** | — | not yet |
| v7b_frbase | v7b, France rows without transformer | 0.98805 | — | not yet |
| **v7c** | v7b; in France the transformer may only LOWER a probability (its France additions looked like look-alikes) | 0.98805 | — | **recommended first upload 26 Sep** |
| v7sib | v3 model + transformers A, B | 0.98780 (tl2 0.98724) | — | not yet |
| v7d | v7c + stage-1 candidates ranked 3–5 rescored | 0.98811 | — | not yet |
| **v7ens** | v7c + e5-base transformer (A2), blended 0.7/0.3 | 0.98813 | **0.983159** (best) | yes, 26 Sep |
| v7ens_frempty | PROBE: v7ens with every France row empty | — | **0.848792** → US/India ~0.9887, France ~0.952 | yes, 26 Sep |
| v7h | v7ens + France descriptor-swap veto | — | — | **no**: refuted on labelled US/India (swaps are 98–99% true) |
| v7i | v7ens, transformer may also ADD France matches | — | **0.982636** → France −0.0035 | yes, 26 Sep |
| v7ens_frbase | PROBE: v7ens, France without transformer | — | — | ready: measures the transformer's France removals |
| v7blend | 0.7 v6-reranked + 0.3 v3-reranked | 0.98801 | — | not recommended |
| v8a | NEW pipeline (lab GPU server): audited blocking + stage-1 GBDT (8% train sample) + 3-fold bge-reranker-v2-m3 on close calls + stack; no stage 2; France min rule + veto | 0.9761 train-sample (not comparable) | — | ready 27 Sep 08:35 (`submissions/v8a`) |
| v8b | v8a, France without the transformer | same | — | ready (`submissions/v8b`) |
| v9 | FULL rebuild with the v8 blocking + stages 1-2 + 3-fold bge + stage 3 (comparable with v7g) | running | — | expected 27 Sep afternoon |
| LB leader | — | — | **0.990621** (26 Sep) | — |

Submissions used on day 1: 5 of 5 (v2, v3, v5, v4, v6). Day 2 so far: v7ens **0.983159** (+0.0132 over v3). Gap to the leader: **0.0075**. Details of the night's work: EXPERIMENTS.md.

---

## 2. Data facts (measured on the files)

| Fact | Value | Consequence |
|---|---|---|
| Train size | S1 2,206,821; S2 5,034,616; S3 5,285,603 | chunked processing; 16 GB laptop |
| Test size | S1 1,732,544 (India 809,986, US 663,106, **France 259,452**); S2 4,887,273; S3 5,082,316 | France has no training labels |
| Singletons (train) | 5.6% | few free points |
| Matches per S1 (train) | mean 3.46 (mostly 2–6; 1–3 from each source) | recall matters too |
| One S2/S3 record matches >1 S1? | never (0 of 7,638,365) | assign each record to at most one S1 |
| Country equal inside matched pairs | 100% | compare only within a country |
| Unmatched S2/S3 records (train) | 26% | decoys exist |
| S1 rows sharing their exact name | 39% (e.g. "Primary Care Group" ×253) | name alone cannot decide; address decides |
| **S2/S3 records per S1** | **train 4.68; test 5.53 (France), 5.82 (India), 5.76 (US)** | test has ~1.9× more unmatched look-alikes per S1 (see §6) |
| Non-Latin names | ~13% of India S2 names, ~7.5% of India S3 (Hindi, Bengali, Odia, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi) | needs cross-script matching |

How matched **names** differ (sample of 207k true pairs): identical after lowercasing 25.5%; only legal words/titles differ 25.2%; words dropped/added 11.7%; very different 10.7% (squashed web domains + invented names ~4%); non-Latin script 7.2%; 1–2 character typo 6.6%; typo + word change 6.2%; clean web-domain form 4.1%; reordered 2.4%; "X d/b/a Name" alias 0.3–1%. Extra noise: injected accents 6.7%, brackets 6.3%, titles (Mr/Sri/M/s) 3.2%.

How matched **addresses** differ: abbreviations/typos 52.7%; state in local script 9.2%; identical 8.4%; very different 7.5%; parts missing 7.4%; house number changed/cut ("103"→"10", "17798"→"0017798") 6.8%; missing 4.4%; reordered 3.6%.

---

## 3. Pipeline as used in v2/v3 (code: `code/business_entity_resolution/src/`)

### 3.1 Cleaning (`normalize.py`, `learn_maps.py`, `prep.py`)
- Names: Unicode NFKD, accents removed, lowercase; web domains turned into words (`exultherbal.com` → `exultherbal`); prefix before `d/b/a`, `a/k/a`, `t/a` dropped; `&` → `and`; punctuation removed; legal words and titles (private, pvt, ltd, llc, inc, corp, co, company, sarl, sas, sa, sci, eurl, gmbh, mr, sri, dr, m/s, the, …) moved out into `name_core`; `name_ns` = core without spaces.
- Addresses: local-script state names mapped to English (14 phrases, learned); accents removed; non-ASCII removed; junk tokens (null, n/a, na, none) removed; 113 short forms expanded (rd→road, st→street, tx→texas, …), **learned from training pairs** (a short token counts if it is a subsequence of a long token that appears only in the other record of a true pair; kept when frequent and dominant); numbers extracted without leading zeros.

### 3.2 Embedding model for non-Latin names (`embed.py`, GPU)
- `intfloat/multilingual-e5-small` (MIT licence, 118M parameters), fine-tuned 1 epoch, in-batch contrastive loss (temperature 0.05), batch 128, lr 5e-5, word-embedding table frozen, gradient checkpointing, max 64 tokens.
- Text: raw name + " | " + cleaned address (first 80 chars). 276,117 training pairs (non-Latin S2/S3 name → S1), **only from S1 rows with crc32(id) % 1000 ≥ 500** (the other half is kept clean for validation).
- Held-out recall@10 among all 883k train India S1: untrained 0.156 → name only 0.573 → **name + address 0.998**.

### 3.3 Shortlist (`candidates.py`), per country, searched from each S2/S3 record
| Search | Method | Top-k |
|---|---|---|
| name | TF-IDF, character 3-grams of `name_ns`, 3-grams in >4,000 S1 rows ignored during search | 10 (30 if the record has no address) |
| address | TF-IDF, word 1–2-grams of cleaned address, terms in >5,000 S1 rows ignored | 10 |
| name + address | TF-IDF, word 1–2-grams of `name_core + address` | 20 |
| embedding | cosine on fine-tuned e5 vectors, non-Latin names only (GPU) | 10 |
- Union of all four ≈ 31 candidates per record; **recall ceiling 0.9851** at full training density.

### 3.4 Features (57 per pair)
TF-IDF cosines (name char-3-gram, address char-3-gram, address words, name+address words; full vocabulary); fuzzy scores (ratio, token-set, token-sort, partial, Jaro-Winkler) on names and addresses; name ratio with legal words kept; house numbers (count, common, first equal / contained / prefix, relative difference, digit edit distance); alphanumeric unit tokens ("8-9-1/14a") overlap; lengths; exact core-name flag; chain count (S1 rows with the same core name); source; flags (web domain, alias, non-Latin name, missing address, non-Latin address); embedding cosine; **competition features within a record's candidates**: margin to the best other candidate and rank for 6 scores, number of candidates, number with name token-set ≥ 90, number with address token-set ≥ 90; which searches found the pair.

### 3.5 Validation design
- 3 folds grouped by true S1 (all records of one business in one fold); unmatched records assigned by id.
- Scored only on the 1,102,482 S1 rows outside the embedding fine-tuning half; models trained only on records whose true S1 is in that half (or unmatched) → no leakage from the embedding.
- Full training density (all 2.2M S1 in the search index).

### 3.6 Stage 1
- Training sample (rows): every matching pair of eligible records + 15% of hard non-matches (rank ≤ 2 by name cosine or address token-set, or name token-set ≥ 80) + 1.5% of other non-matches → 11.47M rows, 3.76M positives.
- LightGBM: 255 leaves, lr 0.08, 600 rounds, min 200 per leaf, feature fraction 0.8, bagging 0.7, L2 1.0. OOF over all 319M training pairs.
- Stage-1 OOF (global threshold 0.925): **0.9724**.

### 3.7 Stage 2
- Input: each record's top 2 stage-1 candidates (2nd only if p1 ≥ 0.01): 10.9M rows.
- Extra features: p1, rank, margin to the other candidate; S1 side: number of other records whose best is this S1 with p1 ≥ 0.5, their sum of p1, best other p1, same-source count, this record's rank among all records pointing at the S1, number of such records.
- **v3 adds sibling features**: for the S1 in question, take the best *other* record assigned to it (the "sibling") and add sibling p1, name token-set(record, sibling), address token-set(record, sibling), first house number equal, same source, sibling missing.
- v2: LightGBM (127 leaves, lr 0.05, 500 rounds). v3: GPU XGBoost (depth 8, eta 0.05, 500 rounds). Trained on 50% of eligible records per fold.
- OOF: v2 0.9788, v3 0.9799.

### 3.8 Decision
- Each S2/S3 record → its single highest-p2 S1. Per S1, among records pointing at it with p2 ≥ 0.3, keep the top-k set maximising expected F0.5 ≈ 1.25·Σp(top k) / (k + 0.25·Σp(all)), versus predicting nothing (Π(1−p)). Beat a single tuned threshold (0.9790 vs 0.9788).

### 3.9 Test run
- Same shortlist on test; stage 1 = one LightGBM trained on the whole sample; stage 2 = one model trained on all eligible rows. Stage-1 test scores are cached (`work/test_base_test_full.parquet`) so stage-2 changes re-predict in ~2 minutes.
- Output check: `check_submission.py` (all rules of the problem statement); every file passed.
- Test output (v2): empty rows France 5.1%, India 5.8%, US 5.8%; mean matches per S1 3.34–3.43; 57–62% of test S2/S3 records assigned (OOF: 70%).

---

## 4. What each step contributed (ablation)

| Step | Validation | Recall ceiling | Macro F0.5 |
|---|---|---|---|
| Hand rule: best (name token-set + address token-set)/200 ≥ 0.8 | 10% slice | 0.987 | 0.8926 |
| v1 LightGBM, 40 features | 10% slice | 0.987 | 0.9771 |
| + embeddings, full-vocab cosines, margins, house-number tokens | 10% slice | 0.996 | 0.9871 |
| + stage 2 | 10% slice | 0.996 | 0.9877 |
| + expected-F0.5 decision | 10% slice | 0.996 | 0.9879 |
| Same design at full density (shortlist bug) | full | **0.797** | 0.848 (stage 1) |
| Fixed shortlist (word-level searches), stage 1 | full | 0.985 | 0.9724 |
| + stage 2 | full | 0.985 | 0.9788 |
| + expected-F0.5 decision = **v2** | full | 0.985 | **0.9790** (LB 0.9677) |
| + sibling features = **v3** | full | 0.985 | **0.9802** (LB 0.9700) |
| Stage 1 as deeper GPU XGBoost (depth 10, 1000 rounds) | full | 0.985 | 0.9720 at stage 1 (vs 0.9724) |
| Training on "test-like" data made by deleting pairs (v4) | test-like (flawed) | 0.984 | 0.9775 there; **LB 0.9614** |
| v3 + consensus features (agreement with all records pointing at the S1) | full | 0.985 | **0.98151** (+0.0014) |
| + calibration (isotonic) and per-country rule | full | 0.985 | 0.98173 (+0.0002; not used, see §7) |
| Self-training on test pseudo-labels | test-like | — | 0.97921 vs 0.97926 without (not used) |

---

## 5. Mistakes made and fixed (so they are not repeated)
1. **Validation slice too small**: a 10% S1 slice hid that the shortlist collapses at full size (recall 0.797 vs 0.996), because fixed document-frequency caps removed city/street 3-grams when S1 is 10× bigger. Fixed with word-level searches; now always validated at full density.
2. **Record-level sampling** kept only 182k positives; switched to row-level sampling (3.76M positives).
3. **CPU instead of GPU**: trees were trained and applied on CPU LightGBM for most of the day. XGBoost 2.0.3 (last version working with the laptop's CUDA 11.6 driver) now runs on the GPU (~8× faster scoring).
4. **Memory**: GPU (4 GB) cannot hold the 11.5M-row sample for one model → use 3 fold models; RAM (16 GB) → never run two big jobs together.
5. **Interrupted runs**: a session restart killed the test build → builds resume per chunk; stage-1 results cached.
6. **Fake test-like data (v4, −0.0086 LB)**: to mimic test's lower US density, `testlike.py` deleted the candidate pairs of dropped S1 rows. Records then had 15.9 candidates (US) instead of the 31.3 they have on test, because a real search refills the list with other S1 rows. The model learned on distorted candidate-count, rank and margin features, and its held-out score could not show it (same distortion). Fix: `testlike_build.py` drops S1 rows **before** the search (tl2, building 25 Sep night). Rule: compare every candidate-list feature with test before training on resampled data.
7. **Two changes in one upload (v6)**: consensus features (US/India) and France cleaning went up together, LB fell 0.0027, and the cause cannot be assigned. v7 (v6 US/India + v3 France) separates them. Rule: one change per upload unless both are validated on data that tracks the LB.
8. **Held-out gains did not reach the LB twice** (v4, v6). The FULL held-out set does not look like test (fewer look-alikes per S1, 2× denser US S1). Rule: until tl2 is shown to rank v2 < v3 and v4 far below, a held-out gain alone is not enough to upload.

---

## 6. OOF vs LB gap (−0.010 to −0.011): analysis so far
- Test has 5.5–5.8 S2/S3 records per S1 vs 4.68 in train, while our predicted matches per S1 on test (3.37) ≈ train truth (3.46) → **test likely has ~1.9× more unmatched look-alikes per S1**.
- Simulation (`decoy_sim.py`): duplicating unmatched OOF records to the test rate lowers v2 OOF 0.9790 → **0.9773** only; re-tuning the decision gains +0.0002. → explains ~0.002 of the gap.
- v3's sibling features (which target look-alikes) gained **2× more on LB than on OOF** → consistent with test having more look-alikes.
- Unexplained ~0.008–0.009. **Main suspect: France** (15% of test, no labels). Arithmetic: France at ~0.92 and US/India at ~0.977 give ~0.968.
- **v5 result (LB 0.838)**: France = 0.132 / 0.1497 + 0.056 ≈ **0.937** (0.934–0.941 with LB rounding); US+India = (0.970 − 0.1497 × 0.937) / 0.8503 ≈ **0.976**. So the gap is ~0.006 France + ~0.004 US/India.

## 6c. France diagnosis (25 Sep 19:00–21:30)
- France is not under- or over-matched: matches per S1 France 3.43, US 3.38, India 3.34 (train truth 3.46). Its uncertain band is not wider (8.5% vs US 8.8%). Its loss is confident mistakes.
- France S1 has many near-twins: same generic name ("<city> <word> SARL") and street, differing only in legal form or house number. Records whose 2nd candidate also has p1 > 0.5: France 2.07%, US 0.39%.
- France S2/S3 text differs from its S1 in ways the US-trained cleaning misses: dotted legal forms (s.a.r.l., ~5% of records), "et" for "&", street short forms (r., av., bd., all., imp., rte., ch., pl.), "N°" before numbers, departement names (Nord, Gironde, Loire-Atlantique) where S1 has region names.
- Checked and ruled out: decoy name words (Holding, Participations…) are already rejected by the model (~1% matched); no Paris/Lyon/Marseille; 5-digit postcodes in 0.4% of addresses; test records are not copies of train S1 businesses (0.01%).
- French cleaning rules written (`normalize.py` french=True, split `testfr`): changes 3.1% of France's matched pairs. In v6 together with consensus features: LB −0.0027 vs v3. v7 tells which of the two caused it.


## 6b. Test-side EDA (25 Sep 18:40) — what differs between train and test
- **Unsure band grows on test in every country**: best-candidate probability 0.05-0.5 = 3.0-3.8% of records in train OOF vs 7.8-8.9% on test (India 7.8, US 8.9, France 8.7). So the gap is not only France.
- The unsure test records are mostly **same name + same street + house number shifted by a few units** (9692 vs 9687 Diamond Rd; 657 vs 652 39th Ave; 441 vs 432 Inspiration Ln).
- In train, such pairs (name token-set >= 85, address token-set >= 70, first house number within 3%, not equal, not a prefix) are **92.7% false** (look-alikes); same-number pairs are 97.1% true.
- Share of records whose best candidate is such a shifted-number look-alike: US **14.3% train vs 21.9% test**, India 2.2% vs 3.2%, France 2.0%.
- Confident pairs (p1 >= 0.9) look the same in train and test (US name/address token-set 93.9/94.9 vs 93.9/94.5; India 77.6/95.3 both) -> test true matches are **not** noisier.
- France confident pairs: 91.8 / 91.0 -> France behaves like the other countries.
- **Density shift**: chain count (S1 rows sharing the core name) of best candidates: US train 22.0 vs test 11.7 (test US S1 is half as dense); India 19.4 vs 17.9. Features tied to density are outside the training range for US.

Implications: (1) the look-alike share is the main measurable difference; (2) US density differs; (3) France does not look worse than the others on any measurable statistic.

---

## 7. Known risks and points to cross-check
1. Stage 2 is trained on **OOF** stage-1 probabilities but applied on test to probabilities from a model trained on all data → possible calibration shift on test.
2. Train S1 density (US 1.32M) differs from test (US 663k); features like chain count, number of candidates and margins may shift.
3. Learned address short forms include US state codes that also hit French words ("de"→"delaware", "la"→"louisiana"). A French-specific cleaning was tried in v6 (result mixed with another change; v7 separates it).
4. France: no labels; legal forms and street words (rue, bd, av) are not in the learned maps.
5. The public LB is a subset of test; its size and France share are unknown (Google Form question).
6. Stage-1 training sample is 33% positives (real rate ~2.5%) → stage-1 probabilities are not calibrated; stage 2 and the decision are tuned on OOF so this should be absorbed.
7. Records with **no address** cause 65% of wrong-S1 errors and half of shortlist misses; chains make many of these ambiguous.
8. Calibration: p2 is under-confident on train (0.45 → 58% true). Calibrating adds ~56k borderline matches on test, in the band where test has 2× train's look-alike share, so train's calibration probably overstates test. Not used until tl2 validates it.

---

## 8. Plan 26–27 Sep: closing the 0.0075 gap to 0.990621 (written 26 Sep 15:00)

### 8.1 Where the gap is (arithmetic, LB = 0.85 × US/India + 0.15 × France)
- The test-like split (tl2) puts v7ens's US/India near **0.9875**. If the LB agrees, France is **(0.983159 − 0.85 × 0.9875) / 0.15 ≈ 0.959**.
- Reaching 0.9906 then needs, for example, US/India 0.991 **and** France 0.986, or US/India 0.993 and France 0.976. Neither country alone can close it.
- The split is an estimate. **Upload `submissions/v7ens_frempty` next**: US/India(v7ens) − US/India(v3) = (LB_probe − 0.838) / 0.85, without assumptions; France follows from 0.983159.

### 8.2 What is left on held-out (after the transformer, EXPERIMENTS.md)
| Loss block | Points | Can we fix it by Sunday? |
|---|---|---|
| true record never in stage-1 top 2 | 0.0068 | 60% are records without address whose name is shared by several S1 rows (no method can pick the right one); the rest needs a new retrieval + full rebuild (≥ 8 h CPU): **no** |
| right S1 found, rejected | 0.0026 | yes, with a better transformer |
| wrong S1 chosen | 0.0011 | yes, with a better transformer |
| wrong records merged | 0.0014 | yes, with a better transformer |

### 8.3 Runs now
1. **Friend's RTX 4060 (8 GB), `docs/FRIEND_RUN.md`:** 3 transformer models (multilingual-e5-base), each trained on 2/3 of all labelled close calls incl. stage-1 ranks 3–5 (1.68M pairs, 2× the old half), scoring every close call out-of-fold and all test close calls. `ce_folds.py`, ~8–9 h. Inputs: `work/bundle_ce_x.zip` (329 MB). Returns 6 score files.
1b. **Friend 2's RTX A6000 (48 GB), `docs/FRIEND2_RUN.md`:** the same 3-fold design with BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M, a pretrained multilingual reranker), fully fine-tuned; ~5–7 h; files sent back after each model.
2. **Kavya's laptop (4 GB):** the same with multilingual-e5-small (`ce_folds.py --name small`, ~6 h, started 14:56) → candidate **v7f**; the friend's e5-base files blended with it → candidate **v7g**. Its held-out result decides whether the 3-fold design beats the 2-halves design before the friend's files arrive.
3. Then: stage 3 on each family, blend (`blend.py`), France rule (`fr_minrule.py`), `finalize.py`. Upload only if held-out (halves protocol) beats v7ens's 0.98813.

### 8.4 France (15% of the score, ~70% of the gap) — checks of 26 Sep 15:40–16:10
- Probe: US/India ~0.9887, France ~0.952 (changes vs v3 are assumption-free: +0.0127 / +0.0158).
- Every France rule is now checked on labelled US/India train first (same data generator), because France has no labels:
  - descriptor swap marks fakes → **refuted** (98–99% true in US/India); v7h not uploaded;
  - legal-form conflict marks fakes → **confirmed** (5.2% / 0.0% true); veto kept;
  - France under-accepts (per-pattern rates far below US) → **refuted by the count bound**: France accepts 861k of ~882k expected true matches (3.4 per S1 as in US/India), so the loss is which records, not how many;
  - France shortlist misses → **small**: 99.05% of same-name same-number pairs are shortlisted (US train: such pairs 99.3% true).
- LB: transformer ADDITIONS in France cost −0.0035 (v7i). Its REMOVALS are untested: `v7ens_frbase` measures them with one upload.
- The friend's scores are used for France with the same setting as v7ens (removals only) unless v7ens_frbase says otherwise.

### 8.5 Honest expectation
- Measured levers: the 3-fold transformer design is untested; earlier transformer variants moved held-out by ≤ 0.0001, the first transformer by +0.0065. A realistic Sunday LB is **0.984–0.986**; 0.9906 needs France near 0.99, which no measured result supports yet.

### 8.6 Night 26→27 Sep (Gathik + Claude Code): blocking audit, deep-learning matcher, v8 — details in EXPERIMENTS.md
- **Blocking audit (laptop, full density):** the name search's df cap (4000, tuned on S10) starves it at full size; with the name cap 10000–20000, word searches min_df 1, no-address k 100 and a dense e5 search for every record, shortlist recall goes US 0.9868 → 0.9971, India 0.9784 → 0.9956 (Latin names). Cleaning bugs: India state codes (pooled map sends 'TN' to 'tennessee', 'DL'/'OD' unmapped; 6.6% of India records) and ordinals ('eleventh' vs '11th'). All of it is opt-in in the code with `BER_V8=1`; defaults are unchanged.
- **Deep learning as the final matcher (lab GPU server, 80k-S1 sample at full density):** fine-tuned bge-reranker-v2-m3 alone 0.99321 vs GBDT 0.98998 on the same candidates; stack 0.99362; the close-call-only design 0.99351. Transformers are the better matcher, the close-call design is right, and the remaining lever is transformer quality. Sample numbers are optimistic (not comparable with 0.98813).
- **v8 (done 27 Sep 08:35 IST):** v8 blocking/cleaning + stage-1 GBDT trained on 8% of the eligible train records + 3-fold bge-reranker-v2-m3 on close calls (37% of test records) + stack, test streamed chunk by chunk. Train-sample held-out GBDT 0.9545 -> v8 0.9761 (record-sampled protocol, not comparable with 0.98813). **v8a** (France: transformer may only lower + legal veto; France 849k matched) and **v8b** (France without the transformer; 887k); US/India identical in both; both pass the official validator. Not validated at FULL density and no stage 2, so v8a changes several things at once vs v7ens: if uploaded, its score tells whether the new pipeline as a whole is better; v8b - v8a then gives the transformer's France effect.

# Appendix A — Problem statement (verbatim, student_resource/README.md)

## ML Challenge 2026 Problem Statement

### Business Entity Resolution Challenge

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as Entity Resolution (ER). Your challenge is to build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

Source 1 is the deduplicated reference source. Your task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many records from Source 2 and Source 3.

#### File Format

**All files in this challenge are tab-separated (`.tsv`), and your submissions must be tab-separated too.** Tabs are used because business addresses and the ID list columns both contain commas. Read them with an explicit tab separator, for example:

```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Reading a `.tsv` without `sep="\t"` will silently produce a single column containing the whole line.

#### Data Description:

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has the following columns:

1. **entity_id:** Unique identifier for the record. The prefix indicates the source — `S1-`, `S2-`, or `S3-`.
2. **business_name:** Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations)
3. **business_address:** Address of the business (may contain partial addresses, format variations, missing components, landmark-based references)
4. **country:** Country label for the record. The **training** data covers `US` and `India`. The **test** set additionally contains a third country, `France`, that does **not** appear in the training data. Treat `country` as an open set of string labels: do **not** hard-code, filter, or one-hot your pipeline to only `{US, India}`, and remember that every test entity — `France` included — must appear in your submission.

There is no separate *source* column — a record's source is given by its `entity_id` prefix (`S1-`/`S2-`/`S3-`) and by which file it appears in.

The ground truth file (`train_ground_truth.tsv`) has two columns:

1. **source1_entity_id:** The `entity_id` of a Source 1 record
2. **matched_entity_ids:** Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 (empty when the entity has no matches)

**Noise Patterns to Expect:**

- **Name variations:** Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs. "and"), word-order transpositions, typos
- **Address variations:** Abbreviations (Rd vs. Road, St vs. Street), transliteration variants, missing components (no PIN code, no state), landmark-based references (Near SBI ATM), municipal numbering formats, component reordering

#### Dataset Details:

- **Training Dataset:** Business records across 3 sources with ground truth matching labels
- **Test Set:** Business records across 3 sources without matching labels

#### File Descriptions:

*Training files*

1. **dataset/train/train_source1.tsv:** Source 1 training records (the deduplicated reference source)
2. **dataset/train/train_source2.tsv:** Source 2 training records
3. **dataset/train/train_source3.tsv:** Source 3 training records
4. **dataset/train/train_ground_truth.tsv:** Ground truth matching labels for the training set

*Test files*

1. **dataset/test/test_source1.tsv:** Source 1 test records. Generate matches for every entity in this file.
2. **dataset/test/test_source2.tsv:** Source 2 test records
3. **dataset/test/test_source3.tsv:** Source 3 test records

No ground truth is provided for the test set. To measure your own performance, hold out a validation split from the training data and score it yourself using the F_0.5 formula given below.

#### Output Format:

Your solution produces **two** tab-separated files, both placed in the `output/`
folder of your final submission package (see *Final Submission Package* below):

1. **`matching_results.tsv`** — your final entity matches. **This is the only file
   scored on the leaderboard** — it is what you upload to the Portal during the challenge.
2. **`candidate_pairs.tsv`** — the candidate set your blocking / candidate-generation
   stage produced, before your final matching model narrowed it down.

##### matching_results.tsv

Your final entity matches:

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| matched_entity_ids | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 |

**Example** (columns separated by a single tab, ID lists separated by commas with no quoting):

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Important:**

- Every Source 1 entity in the test set must have exactly one row
- Leave `matched_entity_ids` empty for entities with no matches (singletons)
- No duplicate entity IDs within a single ID list
- ID lists must only contain Source 2 or Source 3 IDs that exist in the test set

##### candidate_pairs.tsv

The candidate set from your blocking stage — every Source 2 / Source 3 record you
considered a plausible match for each Source 1 entity, *before* your final matching
model narrowed it down. This is the **exact set of records you feed into your matching model
for inference** — the final candidate list *just before* the ML model scores them, not
the raw output of an early blocking pass you later filter further. If your pipeline has
several blocking/filtering stages, `candidate_pairs.tsv` is the *last* one: whatever
your model actually runs inference over. Every ID in `matching_results.tsv` should
therefore appear here.

It is **not scored on the leaderboard**; we use it to analyse blocking quality (recall
ceiling, reduction ratio) and to verify your pipeline.

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| candidate_entity_ids | Comma-separated list of candidate `entity_id`s from Source 2 and/or Source 3 |

**Example:**

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

Same rules as `matching_results.tsv`: one row per Source 1 entity, `candidate_entity_ids`
empty when blocking found no candidates, S2-/S3- IDs only, no duplicates within a list.
Your final matches should be a **subset** of your candidates (a matched ID that never
appeared as a candidate signals a pipeline bug — the validator warns about it).

**Validate before submitting:** a helper script `utils/validate_submission.py` (stdlib
only, no dependencies) checks both files against every rule above so you can catch a
rejection locally instead of spending a submission on it. Run it from this
`student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It prints `PASS` (exit 0) when the files are safe to submit, or a numbered list of issues
to fix (exit 1). It only reads your output files and the test source files; it does not
compute your score.

#### Final Submission Package:

In addition to your live leaderboard uploads, **every team submits a single zip
archive** with your code and outputs. We use it to reproduce your results, audit your
blocking, and check the fair-play and model-license rules — the top teams' packages are
reviewed in detail before the final rankings are confirmed.

Structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
```

- **`output/`** — the two TSV files described above: `matching_results.tsv` and
  `candidate_pairs.tsv`.
- **`code/business_entity_resolution/`** — a self-contained, runnable copy of your
  pipeline. Put all source under `src/`, and include a `README.md` with exact run
  instructions plus a `requirements.txt` (or equivalent environment file) pinning
  versions. Anyone should be able to regenerate both output files from the
  training/test data using only what is in this folder.
- **Methodology document** — fill in the provided `Documentation_template.md` and drop
  it straight into the zip (the filled-in `.md` is fine; a `.pdf` export works too). No
  need to rename it.

#### Constraints:

1. Format your output exactly as described above. Submissions that fail validation will not be evaluated. You should see a `SCORED` status with your F_0.5 score if the output is correctly formatted.
2. `matched_entity_ids` must only reference entities from Source 2 or Source 3. Self-matches to Source 1, and IDs that do not exist in the test set, will be rejected.
3. Every Source 1 entity must appear in your submission. Missing entities will cause rejection.
4. Duplicate entity IDs in any ID list will cause rejection, as will duplicate `source1_entity_id` rows.
5. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

#### Evaluation Criteria:

Submissions are evaluated using **F_β Score (β = 0.5)** — a precision-heavy metric that penalizes false merges (matching two different businesses) more than missed matches.

**Formula:**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed as a **macro-average**: F_0.5 is calculated per Source 1 entity, then averaged across **all** Source 1 entities in the evaluation set.

Singletons are included in that average. A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it. Correctly identifying singletons therefore earns credit, and false merges on them are penalised.

**Why precision-heavy?** In real-world entity resolution, merging two distinct businesses (false positive) is more damaging than missing a link (false negative). F_0.5 weights precision 2× over recall.

**Example:**

- Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
- Ground truth says S1-00001 matches [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0
- F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

#### Leaderboard Information:

- **Public Leaderboard:** During the challenge, rankings will be based on a subset of the test set to provide real-time feedback on your model's performance.
- **Private Leaderboard:** After the challenge ends, the private leaderboard will be revealed, which uses the remaining portion of the test set for evaluation.
- **Final Rankings:** The final decision will be based on the private leaderboard.

You submit predictions for the full test set in both cases; the split is applied during scoring.

#### Submission Requirements:

1. **Leaderboard (during the challenge):** upload `matching_results.tsv` in the Portal —
   tab-separated, with the exact column names described above. This is what drives the
   public and private leaderboards.
2. **Final submission package:** submit the single zip described in *Final Submission
   Package* above — `output/` with **both** `matching_results.tsv` (final matches) and
   `candidate_pairs.tsv` (your candidate-generation / blocking set fed to the model),
   `code/business_entity_resolution/` (runnable pipeline), and your methodology document.
   All teams must submit it; the top teams' packages are reviewed before the final
   rankings are confirmed.
3. Your methodology document must describe:
   - Methodology used
   - Candidate generation / blocking strategy
   - Model architecture and feature engineering
   - Any other relevant information about the approach

   A template for this documentation is provided in `Documentation_template.md`. There is no page limit — prioritise clarity and technical depth over brevity.

#### **Academic Integrity and Fair Play:**

**⚠️ STRICTLY PROHIBITED: External Data Lookup**

Participants are **STRICTLY NOT ALLOWED** to use external databases, APIs, or services to look up business identities or resolve entities. This includes but is not limited to:

- Using commercial entity resolution APIs or services
- Looking up business registrations from government databases
- Using geocoding APIs to normalize addresses
- Any external data augmentation from internet sources

**Enforcement:**

- All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
- Any evidence of external data lookup will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the provided training data.

#### Tips for Success:

- Invest in a strong blocking/candidate generation strategy — it determines the upper bound of your recall
- Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and address matching
- Pay attention to country specific address patterns
- Consider the precision-recall trade-off carefully — F_0.5 rewards precision more than recall
- Do not neglect singletons — correctly predicting "no match" is worth a full 1.0 on that entity
- Validate your own output format against the rules above before submitting

# Appendix B — Guidelines and key instructions (verbatim text of guidelines_and_rules.pdf)

```
We appreciate your participation in the Amazon ML Challenge 2026!!

With the upcoming ML Challenge, we urge you to review the following key instructions and
guidelines meticulously. Your attention to detail and adherence to these guidelines will greatly
contribute to your success in this endeavour.

Prep before you start: Walk through the blog for ML Challenge on best practices and live demo.

Note: Any form of cheating, plagiarism, or unfair practices, such as registering and attempting the
challenge via multiple IDs, will not be tolerated and will lead to instant disqualification of the
participant.

Kindly read through the following vital instructions and important guidelines for this round:

Key Instructions:

      Challenge Window: 25th September 2026, 12:00 AM IST to 27th September 2026, 11:59
          PM IST.

      All teams will get access to the problem statement with the dataset on day 1 and will have
          time to build and submit solutions till day 3.

      Teams can track their performance through the leaderboard, which will reflect team
          rankings live over the course of this challenge. After the challenge, the final leaderboard will
          be revealed.

      Please use this Google Form to ask any queries during the hackathon.
      The below-mentioned artefacts need to be shared for the best solution submitted by the

          team:
                1-2-page document explaining the ML approach, ML models used, experiments and
                    conclusion.
                Source code used for experiments, training and inference, with proper comments
                    describing the functions.

      Each team can make a maximum of 5 submissions per day for over 3 days of the
          hackathon, after which the submit button will be disabled.

      Maintain the version history of all your submissions, as shortlisting will be based on the
          submitted solutions. Participants may also be required to submit the final source code at a
          later stage.

      There will be two leaderboards - Private and Public. Evaluation and shortlisting will be
          based on performance across both leaderboards.

      After successful submission of the artefacts, leaderboard score and each team member
          satisfying the eligibility criteria, the top 100 teams will be announced.

      The Top 100 teams will then be required to submit the following details/documents:
                Methodology used
                Candidate generation/ Blocking strategy
                Model Architecture and feature engineering
                Any other relevant information about the approach.

Simultaneous Logins and Accessibility:

      You can attempt the ML Challenge on a desktop or laptop only and not on a mobile
          device.

      Simultaneous logins are not allowed; i.e. you can only attempt the ML Challenge from one
          laptop or desktop per participant.

      In case simultaneous logins are detected, the system may terminate the ML Challenge
          altogether, and you may only get error messages.

Other instructions:

      If you face any technical problem, clear your browser's cache or try it on a different browser
          or in incognito mode.

      You may also try changing your internet - mobile hotspot, wifi, etc.
      Please shoot an email to support@unstop.com with a screenshot of the page where you are

          facing a problem and your registered email ID. Please note that we won't be helping you
          make decisions, and any email asking us to make decisions will not be entertained.

All the best!

Regards,
Team Amazon ML Challenge 2026
```

# Appendix C — Problem explainer video transcript (verbatim, speech-to-text)

```
[00:00] Welcome to the Amazon ML Challenge 2026.
[00:03] This year's problem is business entity resolution,
[00:06] a fundamental and widely encountered problem
[00:08] in real world data.
[00:10] You will be given business records
[00:11] that arrive from three independent sources,
[00:14] each noisy and inconsistent,
[00:16] and the goal is to determine which of them
[00:18] describe the same real world business.
[00:21] In this video, I will walk you through
[00:22] the problem statement, the data set,
[00:25] the scoring, the submission format,
[00:27] and how entries are judged.
[00:29] Let's start with where this data comes from.
[00:32] Consider a business signing up on Amazon Business.
[00:35] At signup, we capture its core details,
[00:37] such as the business name and address.
[00:40] To build a richer picture of that same business,
[00:42] we pull in additional information from other data providers.
[00:45] Each of these sources typically comes
[00:47] from a different data vendor,
[00:49] with its own formats and conventions.
[00:51] The difficulty is that these external sources
[00:53] share no common identifier with ours,
[00:56] so the only fields we can rely on
[00:58] are the business name and address.
[00:59] For this challenge, we have deliberately limited
[01:02] the data to names and addresses.
[01:05] Here is the core difficulty.
[01:06] The same real world business
[01:08] is described differently by each source.
[01:10] One vendor may write Acme Robotics Incorporated,
[01:13] another abbreviates the address,
[01:15] and a third references a nearby landmark.
[01:18] Source one is our clean,
[01:19] deduplicated reference list.
[01:21] Sources two and three are the noisy fragments
[01:24] that must be reconciled against it,
[01:26] and there is no shared identifier linking them.
[01:29] Entity resolution is the task
[01:31] of linking these records together,
[01:33] establishing that differently written records
[01:35] all refer to the same business.
[01:37] For every entity in source one,
[01:39] the goal is to find all of its matching records
[01:42] in sources two and three.
[01:44] A source one entity may match many records,
[01:46] exactly one, or none at all.
[01:49] Comparing every source one record
[01:51] against every source two and source three record
[01:54] would be far too expensive at scale.
[01:56] So we first apply blocking,
[01:58] sorting records into buckets using a cheap key
[02:00] built from both the name and the address,
[02:03] so that records likely to match land in the same bucket.
[02:06] Each color here is one such block.
[02:08] Notice that every record carries its own id,
[02:11] a noisy name, and an address,
[02:13] and records can group either through a similar name
[02:15] or through a shared address.
[02:17] We follow the orange block,
[02:18] built around Acme Robotics,
[02:20] and the records in it
[02:21] become the candidate matches for that entity.
[02:24] Blocking favors recall,
[02:26] so the bucket also pulls in lookalikes,
[02:28] a business with a similar name at a different address
[02:31] and a different business that happens to share an address.
[02:34] The matching model removes those in the next step.
[02:37] Blocking narrows an enormous number of possible comparisons
[02:40] down to a manageable set of candidate pairs.
[02:43] Finally, a matching model scores each candidate pair
[02:47] and keeps only the true matches, discarding the rest.
[02:50] That produces the two outputs of this challenge.
[02:53] The candidate pairs from your blocking stage
[02:55] and the final matching results,
[02:57] which is the file scored on the leaderboard.
[03:00] You are provided with two datasets.
[03:02] The training set contains records across all three sources
[03:06] together with the ground truth labels,
[03:08] a file that specifies for each source one business
[03:11] exactly which source two and source three
[03:13] records it matches.
[03:15] The test set contains the same three sources,
[03:17] but no labels,
[03:18] and this is what you generate predictions for.
[03:21] A note on the label format.
[03:22] The ground truth is stored as one row per source one entity.
[03:26] It's ID may tee to a comma separated list
[03:29] of all of its matching IDs,
[03:30] so a single row carries that entity's complete match set
[03:34] and the list is empty when the entity matches nothing.
[03:37] This mirrors exactly what you submit.
[03:39] One practical reminder,
[03:41] all files are tab separated,
[03:43] so read them with an explicit tab separator,
[03:45] otherwise the columns will not parse correctly.
[03:49] There are two deliverables.
[03:50] Throughout the challenge,
[03:52] you upload a single file matching underscore results.tsv
[03:56] containing one row per source one entity
[03:58] with its predicted matches.
[04:00] This is the only file scored on the leaderboard.
[04:02] At the close of the challenge,
[04:04] every team also submits a single archive.
[04:07] It contains the final matches
[04:09] along with candidate underscore pairs.tsv.
[04:12] The candidate set your blocking stage
[04:13] produced before the model narrowed it down.
[04:16] This file is not scored,
[04:18] but it is used to audit the quality of your blocking.
[04:20] The archive also includes your complete runnable pipeline
[04:24] and a methodology document describing your approach.
[04:27] The packages of the top teams are reviewed in detail
[04:29] before final rankings are confirmed.
[04:32] One reminder,
[04:33] run the provided validation script before submitting,
[04:36] so a simple formatting error does not cost you a submission.
[04:40] Finally, how entries are judged.
[04:42] Submissions are scored using the macro F0.5 metric,
[04:46] which weights precision twice as heavily as recall.
[04:48] In practical terms,
[04:50] incorrectly merging two different businesses
[04:52] is penalized roughly twice as much as missing a true match.
[04:56] So when in doubt,
[04:57] it is safer not to merge.
[04:58] A few recommendations.
[05:00] First, understand singletons.
[05:02] A singleton is a source one entity
[05:04] that has no matching record in source two or source three.
[05:08] If you correctly predict an empty list for it,
[05:10] you earn a full score of one on that entity.
[05:13] If you predict any match, you score zero.
[05:15] So identifying the businesses with no match
[05:18] is just as important as finding the ones that do match.
[05:21] Second, your blocking strategy sets the ceiling
[05:24] on the recall you can achieve.
[05:26] So invest in it first
[05:27] because you cannot match a record you never consider.
[05:30] And third, pay attention to region-specific patterns
[05:33] in both names and addresses.
[05:35] One firm rule.
[05:36] This is a pure machine learning challenge,
[05:39] so external databases, APIs, and lookups
[05:42] are strictly prohibited.
[05:44] Use only the provided data.
[05:46] That is the challenge.
[05:47] Build something you are proud of,
[05:49] resolve those entities, and enjoy the process.
[05:52] We are excited to see what you create.
[05:54] All the best.```

# Appendix D — Documentation template (verbatim)

## ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

### 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

### 2. Methodology

#### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

#### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

### 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

### 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

### 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

### 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

### Appendix

#### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

#### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

# Appendix E — Experiment log (EXPERIMENTS.md)

## Experiment log

Every run: what changed, the score on held-out training data (CV), and the decision taken.
CV = macro F0.5 on out-of-fold predictions. Never compare scores from different validation setups without saying so.

### Validation setups
- **S10**: 10% of train S1 rows (chosen by id hash) + all their true matches + 10% of unmatched S2/S3 rows. 3 folds grouped by true S1. Fewer same-name competitors than test, so scores are optimistic.
- **FULL**: all 2.2M train S1 rows and all 10.3M S2/S3 rows (same density as test). Introduced in exp 3.

### Runs

| # | Date (IST) | Setup | Change | Recall ceiling | CV F0.5 | Decision |
|---|---|---|---|---|---|---|
| 0 | 25 Sep 01:30 | S10 | Rule only: best (name token-set + address token-set)/200, threshold 0.8 | 0.9870 | 0.8925 | Baseline reference |
| 1 | 25 Sep 01:30 | S10 | LightGBM, 40 features, TF-IDF max_df 5%/2% (relative) | 0.9870 | **0.9786** | Good, but test run stalled on US: relative max_df lets common 3-grams hit ~30k S1 rows each |
| 2 | 25 Sep 02:00 | S10 | TF-IDF max_df absolute 4000/2000 docs (speed fix) | 0.9869 | 0.9753 | Test run 65 min. **Submitted file v1.** Lost 0.003 because the cosine features also lost common 3-grams |

### Error analysis of run 2 (S10 OOF, 25 Sep 04:00) — `src/error_analysis.py`, log `work/errors_v1.log`
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

### Embedding search for non-Latin names (`src/embed.py`, GPU RTX 3050 4 GB)
Metric: recall@10 = true S1 among the 10 nearest of all 883k train India S1 rows, on 20k held-out non-Latin S2/S3 rows
(their S1 is outside the fine-tuning partition).

| # | Model / text | recall@10 | Decision |
|---|---|---|---|
| E0 | multilingual-e5-small, not fine-tuned, name only | 0.156 | Off-the-shelf model does not bridge scripts |
| E1 | fine-tuned (276k pairs, 1 epoch, in-batch contrastive, word-embedding table frozen for 4 GB), name only | 0.573 | Better, but repeated names (chains) cap name-only search |
| E2 | fine-tuned, text = name + " | " + cleaned address (64 tokens, gradient checkpointing) | **0.998** | **Use E2** for blocking + `emb_cos` feature |

GPU lessons: bs 256 with trainable word table OOMs; freeze word table; score queries against 883k S1 in batches of 256 (2048 needs 3.4 GB).

### v2 pipeline on S10 (25 Sep 05:10–06:10)
Eval = all S10 S1 rows (they are all outside the embedding fine-tuning partition, so no leakage). v1 on the same rows: 0.9771.

| # | Change | Recall ceiling | CV F0.5 | Decision |
|---|---|---|---|---|
| 3 | v2 stage 1: + embedding search/feature, full-vocab cosines, margin-to-2nd, ambiguity counts, alnum house-number features, name top-30 when no address; LightGBM 255 leaves, 600 rounds, 16M-row sample | **0.9961** (was 0.9869) | **0.9871** | +0.010 over v1. Keep all |
| 4 | same, sample rate 0.4 (8M rows, to fit RAM) | 0.9961 | 0.9863 | Less data costs 0.0008 -> full data should help |
| 5 | + stage 2 (S1-side competition: p1 margin, other rows' p1 for same S1, same-source counts) | 0.9961 | **0.9877** | +0.0014. Keep. Per-segment thresholds: +0.0000 -> turned off (slow) |

Error analysis of run 5 (`work/s10_errors.log`), points lost: wrong_s1 0.0045 (73% have NO address; chain names like "Perfect Food Pvt Ltd" at several S1 addresses), under_thresh 0.0043 (34% no address), decoy_merge 0.0036 (look like true matches: "Consolidated Médical Studios | 31 Laurel Circle" vs S1 "... | 315 Laurel Circle", p=0.99; likely irreducible), blocking_miss 0.0014 (31% no address). Non-Latin names are no longer a problem (0.2% of misses vs 60% in v1).

### FULL setup (all 2.2M train S1, 10.3M S2/S3; eval = 1.1M S1 outside the embedding partition)
Build: 42 chunks, ~100–130 s each (~85 min), 9.1 GB of features on disk.

| # | Change | Stage-1 sample | Result | Decision |
|---|---|---|---|---|
| 6 | Record-level 6% sample (SAMPLE_RATE) | 7.2M rows, only **182k positives** | stopped before training | Record sampling throws away 94% of positives. Switched to row-level sampling |
| 7 | Row-level: all eligible positives + 15% hard negatives (rank ≤ 2 by name cosine or address token-set, or name token-set ≥ 80) + 1.5% other negatives; stage 2 on 50% of records | 8.57M rows, 3.04M positives | **recall ceiling 0.7968, stage-1 F0.5 0.8482** | Blocking collapses at full density (US 0.742, India 0.879). Stopped. |

**Root cause (exp 7):** the absolute 3-gram df caps (4000 name / 2000 address) introduced in run 2 for speed are 10x more aggressive at full density than on S10: at 1.3M US S1 rows, every city/street 3-gram exceeds 2000 docs and is dropped. S10 validation hid this (its S1 index is 10x smaller). **Test has full density (US 663k, India 810k S1), so v1 on test was also hurt.** Lesson: validate blocking at the same S1 density as test.

#### Blocking at full density (`src/blocking_exp.py`, 50k held-out US rows vs all 1.32M US S1)

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

#### Run 9 — FULL with the new blocking (25 Sep 08:52–12:55)
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

### Submissions plan 25 Sep (each upload tests one thing; saved as submissions/vN + git tag vN)
| Ver | Changes only | OOF (FULL) | Question the leaderboard answers |
|---|---|---|---|
| v2 | two-stage LightGBM, word-level shortlist, expected-F0.5 | 0.97904 | **LB 0.9677** (OOF - 0.011). OOF does not track LB 1:1 |
| v3 | + sibling-agreement stage-2 features (GPU XGBoost stage 2) | 0.98015 (+0.0011) | **LB 0.9700 (+0.0023)**: gain on test is 2x OOF gain -> decoy/group handling is the lever |
| v4 | stage 1 as deeper GPU XGBoost (depth 10, 1000 rounds) | running | is the model capacity-limited? |
| v5 | France-only threshold shift on best model | - | is France over/under-matched? |
| v6 | average LightGBM + XGBoost scores | - | do two model types add accuracy? |

#### LB gap analysis (25 Sep 17:40)
- Test has 5.5-5.8 S2/S3 rows per S1 vs 4.68 in train; predicted matches per S1 on test (3.37) ~ train truth (3.46) -> test has ~1.9x more unmatched look-alike rows per S1.
- `decoy_sim.py`: duplicating unmatched OOF rows to the test rate drops v2 OOF 0.9790 -> 0.9773 only; re-tuned rule gains +0.0002. So decoy rate explains ~0.002 of the 0.011 gap.
- Remaining ~0.009 unexplained: prime suspect France (15% of test, no labels): France F~0.92 with US/India ~0.977 would give 0.968.

#### Test-side EDA (25 Sep 18:40) — what differs between train and test
- **Unsure band grows on test in every country**: best-candidate probability 0.05-0.5 = 3.0-3.8% of records in train OOF vs 7.8-8.9% on test (India 7.8, US 8.9, France 8.7). So the gap is not only France.
- The unsure test records are mostly **same name + same street + house number shifted by a few units** (9692 vs 9687 Diamond Rd; 657 vs 652 39th Ave; 441 vs 432 Inspiration Ln).
- In train, such pairs (name token-set >= 85, address token-set >= 70, first house number within 3%, not equal, not a prefix) are **92.7% false** (look-alikes); same-number pairs are 97.1% true.
- Share of records whose best candidate is such a shifted-number look-alike: US **14.3% train vs 21.9% test**, India 2.2% vs 3.2%, France 2.0%.
- Confident pairs (p1 >= 0.9) look the same in train and test (US name/address token-set 93.9/94.9 vs 93.9/94.5; India 77.6/95.3 both) -> test true matches are **not** noisier.
- France confident pairs: 91.8 / 91.0 -> France behaves like the other countries.
- **Density shift**: chain count (S1 rows sharing the core name) of best candidates: US train 22.0 vs test 11.7 (test US S1 is half as dense); India 19.4 vs 17.9. Features tied to density are outside the training range for US.

Implications: (1) the look-alike share is the main measurable difference; (2) US density differs; (3) France does not look worse than the others on any measurable statistic.

#### Fable review checks (25 Sep 19:10)
- Check 3, ID leak: Spearman(S1 id number, matched id number) = 0.0003; sibling S2 ids are ~165M apart (median). No leak.
- Check 5, per-source cap: matches per (S1, source) = 1: 1.51M, 2: 1.32M, 3: 0.71M, 4: 0.26M, 5: 59.5k, 6: 2.8k. No useful cap.
- Check 6, calibration of v3 p2 (best candidate per record): under-confident in the middle (predicted 0.45 -> actual 0.56; 0.15 -> 0.23; 0.86 -> 0.92). Isotonic calibration before the expected-F0.5 rule is worth testing (calibrate on test-like OOF).
- Check 4 (test look-alikes = train businesses?): pending, needs RAM after v4 training.
- Forum rules (25 Sep): hand-written normalisation dictionaries, unsupervised stats on test, self-training allowed; libpostal/gazetteers/APIs not allowed. Plan in docs/AWS_PLAN.md.

#### v5 result and France diagnosis (25 Sep 19:00-19:30)
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

#### v4 = training on test-like conditions (25 Sep 18:59-19:25)
`testlike.py`: US S1 kept at test density (50%), India 91.7%; look-alike records topped up to test's records-per-S1 (US 5.76, India 5.10). Same features, GPU XGBoost both stages + sibling features.
- Held-out on the test-like setup: stage 1 0.9719, stage 2 0.9775, recall ceiling 0.9844; best rule plain threshold 0.55 (expected-F0.5 rule 0.9759 there). Not comparable with v3's 0.9802 (different validation data).
- France cleaning rules written (`normalize.py` french=True, `prep.py france` -> split `testfr`), not yet used.

#### v6 ingredients (25 Sep 20:00-20:25), all on the test-like validation (v4 base 0.97753)
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

#### v4 LB 0.961357 (25 Sep 20:50) — test-like training HURT (-0.0086 vs v3)
- Cause (checked on data): `testlike.py` removed S1 rows by deleting their candidate pairs, so records kept only the
  remaining candidates. Candidates per US record: full train 31.4, **test-like 15.9**, test 31.3; candidates with name
  token-set >= 90: full 3.4, test-like 1.87, test 3.84 (India 30.1 / 27.5 / 30.1). The real search at lower density
  refills each record's top-k with other S1 rows, so test looks like FULL train on these features, not like test-like.
- The test-like held-out score (0.9775) could not see this: its validation rows have the same distortion.
- Decision: drop test-like training. v6a (built on it) not uploaded. v6 rebuilt on FULL data (v3 base) + consensus
  features + calibration + France cleaning.

#### v6 on FULL data (25 Sep 20:50-21:40)
| Change (FULL validation, same as v3) | Held-out macro F0.5 | Decision |
|---|---|---|
| v3 (reference) | 0.98015 | LB 0.9700 |
| + consensus stage-2 features (LightGBM stage 1 of v3, XGBoost stage 2) | 0.98151 (+0.0014); `cons_num_share` is the 3rd most important stage-2 feature | **v6** |
| + calibration, per-country rule | 0.98173 (+0.0002 more) | left out: on test it adds ~56k borderline matches (p2 0.4-0.5), the band where test has 2x train's look-alike share |
| v6 = consensus + v3's rule (expected-F0.5, floor 0.3, raw p2) + France from the cleaned rebuild | - | 1,635,015 S1 rows with matches; matched records vs v3: US 2.261M (+20k), India 2.746M (+39k), France 0.908M (+19k) |

### Lessons (read before changing anything)
- **Resampled training data must be checked against test on every feature that depends on the candidate list** (candidates per record, counts, ranks, margins) before training on it. Deleting pairs is not the same as searching at lower density (v4: -0.0086 LB).
- GPU 4 GB cannot fit XGBoost on the full 11.5M-row stage-1 sample (OOM after 3 folds, v4 first try). Use fold models (saved immediately) and average them for test.
- A Claude Code session restart kills background jobs. Every long step must be resumable: build skips chunk files already on disk (added 25 Sep 13:45 after the test build died at India chunk 6); stage 1 and test stage-1 scores are cached.
- A relative `max_df` makes search cost grow with S1 size; use an absolute document-frequency cap for **search**, but compute similarity **features** with the full vocabulary.
- Always time one chunk at test scale (US test S1 = 663k rows) before launching a full run.
- Prediction share per country on test (v1): India 12.8% empty vs 5.6% singletons in train, so India is under-matched.

# Appendix F — Submission log (submissions/LOG.md)

## Submission log

| Version | File | Time (IST) | CV macro F0.5 | Public LB | What changed |
|---|---|---|---|---|---|
| v1 | `v1_matching_results.tsv` | 25 Sep 03:02 | 0.9753 (10% train slice, 3-fold) | not submitted (team decision: broken full-density shortlist) | TF-IDF name+address shortlist (top 10 each, same country), 40 features, LightGBM, each S2/S3 row to its best S1 if prob >= 0.75 |
| v2 | `submissions/v2/` | Fri 17:00 | 0.97904 (FULL OOF) | **0.9677** | two-stage LightGBM, word-level shortlist, expected-F0.5 decision (1633975 S1 rows with matches) |
| v3 | `submissions/v3/` | Fri 17:15 | 0.98015 (FULL OOF) | **0.9700** | v2 + sibling-agreement features in stage 2 (GPU XGBoost stage 2) (1634819 S1 rows with matches) |
| v5 | `submissions/v5/` | Fri 17:35 | 0.98015 (FULL OOF) | **0.838** -> France ~0.937, US+India ~0.976 | PROBE: v3 with all France rows emptied (measures France score) (1388739 S1 rows with matches) |
| v4a | `submissions/v4a/` | Fri 18:49 | 0.98005 (FULL OOF) | _upload & fill in_ | stage 1 as deeper GPU XGBoost (depth 10, 1000 rounds) + v3 stage 2 (1634872 S1 rows with matches) |
| v4 | `submissions/v4/` | Fri 19:56 | 0.97753 (test-like OOF, not comparable to FULL) | **0.961357** (-0.0086 vs v3: test-like data halved US candidate lists, 15.9 vs 31.3 on test) | trained on test-like conditions (US S1 at test density, look-alike records topped up) + sibling features, GPU XGBoost (1633498 S1 rows with matches) |
| v6a | `submissions/v6a/` | Fri 20:46 | 0.97966 (test-like OOF, calibrated) | NOT UPLOADED (built on v4's test-like models) | v4 + consensus stage-2 features + calibrated per-country decision (held-out 0.97966 vs v4 0.97753, test-like validation) + France rows from French-cleaned rebuild (1637900 S1 rows with matches) |
| v6 | `submissions/v6/` | Fri 21:32 | 0.98151 (FULL OOF, same validation as v3) | **0.967332** (-0.0027 vs v3; consensus and France cleaning not separable from this upload) | v3 base (full data) + consensus stage-2 features (held-out 0.98151 vs v3 0.98015) + France rows from French-cleaned rebuild; v3's decision rule, no calibration (1635015 S1 rows with matches) |
| v7 | `submissions/v7/` | Fri 21:40 | 0.98151 (FULL OOF) | _upload & fill in_ | PROBE: v6 US/India (consensus features) + v3 France rows -> LB minus 0.9700 = consensus effect on US/India; v6 minus this = France cleaning effect (1634670 S1 rows with matches) |
