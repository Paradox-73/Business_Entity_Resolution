# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Mommy's Good Boys  
**Team Members:** Kanav Bhardwaj, Bhavya Jain, Harsh Gupta, Gathik Jindal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
Each Source 2 / Source 3 record is searched against the Source 1 (S1) records of its own country with four top-k searches (character and word TF-IDF, a fine-tuned multilingual embedding), a LightGBM model keeps each record's best few S1 candidates (6.73 per S1 entity on test), and an XGBoost second stage adds features that compare all records pointing at the same S1 entity. Records whose decision is uncertain are re-read by fine-tuned cross-encoders (multilingual-e5-small and bge-reranker-v2-m3, 3 fold models each) whose scores an XGBoost stage 3 per model family combines with the GBDT scores, and an expected-F0.5 rule picks each entity's match set. France, which has no labels, uses the same GBDT stages with an earlier cross-encoder set, a rule that lets the cross-encoders only lower France scores, a legal-form veto, and three pair-level rules checked with a label-free test built on an artifact of the data generator. A second run of the same pipeline with wider blocking (by teammate Gathik) adds 16,421 US/India matches that our candidate lists never contained. Held-out macro F0.5 on US/India is 0.98950 and the public leaderboard score is 0.989563.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the provided files (train: 2,206,821 S1 rows, 10,320,219 S2/S3 records; test: 1,732,544 S1 rows, 9,969,589 records).

- **One entity per record.** Every matched train record belongs to exactly one S1 entity (7,638,365 true pairs = 7,638,365 matched records), always in the same country. 26% of S2/S3 records match nothing (decoys). 5.6% of S1 entities are singletons; the mean is 3.46 matches per entity.
- **Chains.** 39% of S1 rows share their exact name with another S1 row ("Primary Care Group" appears 253 times), so the name alone cannot decide.
- **Name noise in true pairs** (sample of 207k true pairs): identical after lowercasing 25.5%; only legal form or title differs 25.2%; words dropped or added 11.7%; typos 12.8% (1-2 characters 6.6%, typo plus word change 6.2%); clean web-domain form ("exultherbal.com") 4.1%; non-Latin script (Hindi, Bengali, Odia, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi) 7.2%; very different names 10.7% (squashed web domains and invented names ~4%); "X d/b/a Name" aliases 0.3-1%.
- **Address noise in true pairs:** short forms and typos 52.7%; state written in a local script 9.2%; parts missing 7.4%; house number changed or truncated 6.8%; address missing 4.4%.
- **Test differs from train.** Test has 5.75 S2/S3 records per S1 row against 4.68 in train, i.e. more unmatched look-alikes. Records whose best candidate is a same-name, same-street entity with a house number a few units away: US 14.3% of train records, 21.9% of test records; in train such pairs are 92.7% false. US test S1 is half as dense as US train S1 (663,106 vs 1,323,633 rows).
- **France (15.0% of test S1, 259,452 rows) has no training labels.** Its look-alikes are twins with generic names ("<city> <word> SARL") on the same street that differ only in legal form (SA / SAS / SARL / SASU), a descriptor word or the house number.
- **Generator artifacts (measured on labelled US/India train).** Decoy records (records that match nothing) usually move the house number up against their look-alike S1 row (US decoys: +1: 7.3%, +2..5: 29%, +6..20: 30%; US true pairs 0.2-0.3% per band). True records with a changed word are almost never all-lowercase (0.2-0.4%), while decoys are lowercase at the normal rate (2-4%). This gave a label-free way to estimate the true share of any France pair set.
- **Validation must match test density.** A 10% validation slice hid a blocking failure: search document-frequency caps tuned on it dropped every common 3-gram at full density, and full-density recall fell to 0.7968. Every blocking number below is measured with all train S1 rows of the country in the index.

### 2.2 Solution Strategy

**Approach Type:** Blocking + cascade of classifiers (GBDT stage 1 -> GBDT stage 2 with group features -> cross-encoder transformers on uncertain records -> GBDT stage 3) + expected-F0.5 set selection, with a second run of the same pipeline under wider blocking added for recall.  
**Core Innovation:** (1) group features: each record is compared with the other records that point at the same S1 entity (a true match agrees with them, a look-alike does not); (2) transformers applied only to the 23.5% of test records whose decision is uncertain, trained as 3 fold models so that every labelled close call has an out-of-fold score; (3) France rules chosen with a label-free test (the lowercase artifact) and checked with France-only leaderboard submissions.

Validation ("FULL" setup, used for every held-out number unless marked otherwise):
- Train S1 rows are split into halves by crc32(id). The **evaluation half** (~1.1M S1 rows) is scored; the other half is the only one used to fine-tune the embedding model, so the embedding feature cannot leak.
- Candidate search for the evaluation records runs against **all** train S1 rows of the country (same density as test).
- Stage 1, stage 2 and stage 3 are trained with 3 folds grouped by true S1 entity (all records of one business in the same fold) and scored out-of-fold.
- Score: macro F0.5 over the evaluation-half S1 rows, singletons included, exactly as the metric.

---

## 3. Candidate Generation (Blocking)

Every search runs **from each S2/S3 record towards the S1 rows of the same country** and returns a fixed number of candidates, so no pair of records is ever compared exhaustively. Searches are sparse matrix products with a per-row top-k (`sparse_dot_topn`), run in chunks of 100k-250k records against one country's S1 index. Terms shared by more than a fixed number of S1 rows are ignored during the search, which bounds the work per record as S1 grows; the character 3-gram similarity features of section 4 use the full vocabulary, while the two word-level cosines reuse the capped search vectors.

Union of four searches:
1. **Name:** TF-IDF over character 3-grams of the space-free core name (accents stripped, legal forms and titles removed, web domains turned back into words, "d/b/a" aliases dropped); top 10, or top 30 when the record has no address; 3-grams in more than 4,000 S1 names ignored.
2. **Address:** TF-IDF over word 1-2-grams of the cleaned address (short forms such as rd -> road and local-script state names mapped with tables learned from train matches); top 10; terms in more than 5,000 S1 rows ignored.
3. **Name + address:** TF-IDF over word 1-2-grams of core name + address; top 20 (terms in > 5,000 S1 rows ignored). For US and India test records a wider version is used: top 40, terms in > 20,000 S1 rows ignored.
4. **Embedding (non-Latin names only):** multilingual-e5-small fine-tuned with an in-batch contrastive loss on 276k (non-Latin record, S1) train pairs from the non-evaluation half, text "name | address", 64 tokens; exact top 10 by cosine on the GPU. Held-out recall@10 against all 883k India S1 rows: 0.156 before fine-tuning, 0.573 fine-tuned on names only, 0.998 fine-tuned on name + address.

**Learned filter (stage 1).** A LightGBM model scores every blocked pair with 55 inexpensive features (section 4) and keeps, per record, the best candidate and the second one if its probability p1 >= 0.01. For records whose decision is uncertain ("close calls": best p2 in [0.01, 0.995], or second candidate p1 >= 0.2) the stage-1 candidates ranked 3-5 with p1 >= 0.005 are kept as well. Stage 2 scores the top 1-2 candidates; the cross-encoders and stage 3 score the close calls' top 2 plus their ranks 3-5. These surviving pairs, plus the 16,421 second-generator pairs described below, are exactly the pairs of `candidate_pairs.tsv`.

- **Blocking keys used:** character 3-gram TF-IDF of the core name; word 1-2-gram TF-IDF of the address and of name + address; fine-tuned multilingual-e5 embeddings of "name | address" for non-Latin names; same-country constraint; followed by the stage-1 LightGBM top-candidate filter.
- **Candidate pairs generated:** **11,658,294 pairs = 6.73 per S1 entity** (US 6.29, India 6.70, France 7.94; median 6; 377 S1 rows with none), 1.17 per S2/S3 record. Of these, 16,421 are pairs from the second generator (below) that the final file matched. Before the stage-1 filter the four searches return 469.8M pairs on test (47.1 per record, 271 per S1 entity). Against all same-country S1 x record pairs (6.72 x 10^12) the candidate file is a reduction of 99.99983%.
- **How you ensured true matches were not lost:**
  - Each search was measured separately at full density on held-out records (50k US records against all 1.32M US S1 rows):

    | Search | Recall | Search time per 50k records |
    |---|---|---|
    | name char 3-grams, 3-gram cap 4,000, top 10 | 0.555 | 6 s |
    | address char 3-grams, cap 2,000 (first version) | 0.412 | 2 s |
    | address word 1-2-grams, cap 5,000, top 10 | 0.894 | 3 s |
    | name + address word 1-2-grams, top 20 | 0.982 | 5 s |
    | union used (name + address + combined) | 0.9856 (India 0.9593 before the embedding search) | ~14 s |

  - Held-out recall of the whole shortlist (evaluation half, full density): **0.9851** (the first version, with character 3-gram address search, had 0.7968). The embedding search removed non-Latin names as a miss cause (on the 10% slice: 60% of shortlist misses before it, 0.2% after).
  - The stage-1 filter keeps 97.86% of true pairs as the top 1-2 candidates; the ranks 3-5 of close-call records add back 13,569 true pairs on train.
  - Wide name + address search for US/India test records: on labelled train samples it finds 45% of the with-address shortlist misses, and stage 1 ranks 94% of those first. On test, 13.8k of the US/India pairs it added after stage 3 have an S1 entity the production lists never had.
  - Second candidate generator (teammate Gathik's run on a lab GPU server, `server/run_v9.sh`): the same code with `BER_V8=1`, which sets the name 3-gram cap to 10,000, takes the top 100 name candidates for records without an address, keeps every term in the word searches, adds India state-code and ordinal cleaning, and runs the fine-tuned e5 top-10 search for every record. On the same held-out protocol (evaluation half, full density) its shortlist recall is 0.9963 at 40.5 candidates per record (production: 0.9851 at 30.9). Its matcher is the same cascade (XGBoost stage 1, stage 2 with group features, 3 bge-reranker-v2-m3 fold models trained on 1M close-call pairs each, stage 3). The final file takes 16,421 of its US/India matched pairs: pairs absent from our candidate lists, for records our file leaves unmatched. Only those matched pairs (not its full candidate set) are included in `candidate_pairs.tsv`.

---

## 4. Matching Model

**Features used:**
- Name features: TF-IDF cosine over character 3-grams (full vocabulary); edit-distance ratio (rapidfuzz `ratio`), token-set, token-sort and partial ratios; Jaro-Winkler; ratio with legal forms kept; exact core-name flag; embedding cosine (non-Latin names); name lengths; number of S1 rows sharing the core name (chain size); flags for web-domain names, "d/b/a" aliases and non-Latin script.
- Address features: character 3-gram cosine, word 1-2-gram cosine, name + address word cosine; ratio, token-set and partial ratios; house numbers (count on each side, common count, first number equal / contained / prefix, relative difference, digit edit distance); overlap of alphanumeric unit tokens such as "8-9-1/14a"; address length; missing-address and non-Latin-address flags.
- Other:
  - Competition among a record's candidates: margin to the best other candidate and rank, for name cosine, address cosine, name and address token-set, embedding cosine and name + address cosine; number of candidates; number of candidates with name or address token-set >= 90; which searches found the pair; source (S2 or S3). 55 features in stage 1.
  - Stage 2 (76 features = 55 + 21): p1, its rank and margin; S1-side competition (how many other records point at this S1 with p1 >= 0.5, overall and from the same source, their sum and maximum, this record's rank among them); sibling features (the best other record of the same S1: its p1, name and address token-set against this record, same house number, same source); consensus features (all records with p1 >= 0.5 at this S1: mean and minimum name and address token-set, share and count with the same first house number).
  - Stage 3: p1, p2, the cross-encoder logit, its margin and rank among the record's close-call candidates, p2 margin, number of close-call candidates, stage-1 rank.
  - Cross-encoder input: lowercased "name | address" of the record and of the S1 row (first 160 characters each), 128 tokens.
  - France only (no labels; each rule's evidence in brackets):
    - Legal-form veto: p = 0 when both raw names carry a legal form (SARL, SAS, SA, EURL, ...) and none is shared [such conflicts are true in 5.2% of US and 0.0% of India labelled pairs].
    - Min rule: the transformers may only lower a France probability, p = min(stage-2 p2, stage-3 p) [allowing additions cost France -0.0035 on the leaderboard].
    - Descriptor-word veto, 22,436 accepted pairs set to p = 0: the record adds or swaps in a descriptor word (amicale, comite, ecole, club, centre, ...), synonym and stem swaps excluded [3.75% of these records are all-lowercase, against 3.38% for France records with a pure decoy word and 0.2-0.4% for true changed-word records in US/India labels, so ~0% are true].
    - Noise-word additions, 9,800 rejected pairs raised to p = max(p, 0.95): same street and house number, and the record adds or swaps in a noise word, or groupe / developpement / france where the v7m probe had restored the pair [lowercase test ~0.96 true; a fit to the France leaderboard results gave 0.77-0.81].
    - False-accept veto, 1,145 accepted pairs set to p = 0, in three groups: 846 with a legal form added and the house number moved up by 1-20, the decoy signature [lowercase test ~0 true]; 232 with a word added, swapped or mistyped and the house number moved up by 1-20 [lowercase estimate ~0.60 true]; 67 all-lowercase records with a name change, in change patterns whose US/India true records are almost never lowercase [per-pair estimate 0.42 true on average].

**Model type:**
- Stage 1: LightGBM, 255 leaves, learning rate 0.08, 600 rounds, trained on 11.47M sampled pairs (all 3.76M positives, 15% of hard negatives, 1.5% of the other negatives), 3 fold models for out-of-fold scores and one full-sample model for test.
- Stage 2: XGBoost (GPU), depth 8, learning rate 0.05, 500 rounds, on each record's top candidates (50% of the records per training fold); the 3 fold models are averaged on test.
- Cross-encoders on close calls (2,523,745 labelled train pairs; 3,788,098 test pairs in 2,341,622 records): 3 fold models per family, each trained on the close calls of the other two folds (1.68M pairs), one epoch, binary cross-entropy, AdamW with one-cycle schedule. multilingual-e5-small (MIT, 118M parameters): word-embedding table frozen, gradient checkpointing, batch 32, learning rate 3e-5 (4 GB laptop GPU). bge-reranker-v2-m3 (Apache-2.0, 568M parameters): full fine-tuning, batch 64, learning rate 2e-5 (48 GB GPU).
- Stage 3: XGBoost, depth 6, learning rate 0.05, 300 rounds, per family, on the out-of-fold cross-encoder scores; on test the three fold models' scores each go through stage 3 and are averaged. Final probability for US/India: 0.3 x e5-small family + 0.7 x bge family (weights chosen on held-out); records that are not close calls keep the stage-2 p2.
- France rows use the stage-3 blend of an earlier transformer set (0.7 x two e5-small models trained on the two S1 halves + 0.3 x one multilingual-e5-base model, MIT, 278M parameters, trained on 420k close calls), the setting whose France score was measured on the leaderboard (v7ens, France ~0.952 before the France rules).

**Threshold selection method:** expected-F0.5 set selection. Each record goes to its highest-probability S1 entity. For each entity, the records with p >= floor are sorted by p and the set of the top k is chosen that maximises 1.25 x (sum of the top-k p) / (k + 0.25 x sum of all p); the entity gets an empty list when the probability that none matches (product of 1 - p) is higher. The rule was chosen among 17 candidates (thresholds 0.4-0.9, floors 0.3 / 0.5 with exponents 1 / 1.5 / 2 on p) by held-out macro F0.5: floor 0.5, exponent 1.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98950** held-out (US/India, evaluation-half train S1 at full density, all close calls rescored; 0.98878 when records whose close calls span both S1 halves keep their GBDT score, the protocol used for earlier versions). Public leaderboard of the final file: **0.989563**. The held-out number does not include the wide test search, the second generator's pairs or France (no labels).

| Version | Change | Held-out macro F0.5 | Public LB |
|---|---|---|---|
| rule | best name + address token-set >= 0.8 | 0.8925 (10% slice) | - |
| v1 | LightGBM, 40 features | 0.9753 (10% slice) | not submitted |
| v2 | word-level blocking valid at full density, two-stage LightGBM, expected-F0.5 rule | 0.97904 | 0.9677 |
| v3 | + sibling features, XGBoost stage 2 | 0.98015 | 0.9700 |
| v6 model | + consensus features | 0.98151 | 0.967332 (v6, uploaded together with French cleaning) |
| v7 | + e5-small cross-encoder on close calls, stage 3 | 0.98794 | - |
| v7ens | + second half model and e5-base, blend; France: min rule + legal-form veto | 0.98813 | 0.983159 |
| v7p | 3-fold e5-small + 3-fold bge, blend 0.3/0.7; wide US/India test search | 0.98950 (0.98878 halves protocol) | - |
| **v9y (final)** | v7p US/India + 16,421 second-generator pairs; France: v7ens + descriptor veto + noise-word additions + false-accept veto | same | **0.989563** |

France effects were measured with submissions that changed only the France rows: public-LB difference divided by France's share of S1 rows (0.15). Emptying France (v5: 0.838; v7ens_frempty: 0.848792) put France at ~0.937 (v3) and ~0.952 (v7ens) and US/India at ~0.9887 (v7ens). Changes that lost on France: letting the transformer add France matches (-0.0035), adding France records with a changed house number (-0.0163), restoring same-address descriptor swaps (-0.0091). From v7ens (0.983159) to the final file (0.989563) the public score rose by 0.0064. No upload isolates the France part of that rise. The measured or estimated US/India gains (held-out +0.00065 on the halves protocol for the 3-fold e5-small and bge families, +0.0011-0.0015 estimated for the wide search) account for about 0.85 x 0.002 = +0.0017; the second generator's pairs were not measured separately. If they added nothing, France rose by about (0.0064 - 0.0017) / 0.15 = +0.031, from ~0.952 to ~0.98 (estimate).

- **Common false positives (wrong merges):**
  - Look-alike records: same name and street, house number moved up by a few units ("9692 Diamond Rd" vs S1 "9687 Diamond Rd"; "Consolidated Médical Studios, 31 Laurel Circle" vs S1 "..., 315 Laurel Circle"). On held-out after the e5-small transformers (v7b scores, 0.98805; `error2.py`; each number = macro F0.5 points recovered if that mistake type alone were fixed), accepted records that match no S1 cost 0.00083 (2,999 pairs) and accepted records of another S1 entity 0.00060 (2,285 pairs).
  - France: a descriptor word swapped at the same address ("Didier Club SARL" vs "Didier Ecole SARL"). In US train labels such swaps are 98.9% true, in France they were ~40% true (leaderboard result of v7m), so the US-trained models were over-confident on France name changes; the descriptor veto removes 22,436 such pairs (lowercase test: ~0% true).
  - France legal-form change (SARL vs SAS on otherwise equal records); handled by the legal-form veto.
- **Common false negatives (missed matches)** (same held-out analysis as above):
  - True S1 not among the stage-1 top 2: 0.00681 (81,531 pairs; records without an address 0.00416 of it). Of the 3,815,794 held-out true pairs, 1.49% (~57k) are never shortlisted (51% of shortlist misses have no address, run-9 analysis) and 0.65% (~25k) are shortlisted but ranked below 2nd by stage 1; the stage-1 ranks 3-5 of close-call records were added afterwards (v7d onward) for this reason.
  - Right best candidate rejected by the decision rule: 0.00258 (31,933 pairs), largely records with a changed or missing house number (before the transformers: changed number 0.00290 and no number 0.00196 of this mistake type's 0.00692).
  - Best candidate is another S1 entity: 0.00113 (13,842 pairs), 0.00093 of it records without an address whose name is a chain with several S1 addresses ("Perfect Food Pvt Ltd").

---

## 6. Conclusion
A bounded top-k blocking with a learned filter keeps 6.73 candidates per S1 entity while a GBDT cascade with group features and fold-trained cross-encoders on the uncertain 23.5% of records reaches 0.98950 held-out macro F0.5 on US/India and 0.989563 on the public leaderboard. The largest gains came from measuring blocking at test density (shortlist recall 0.7968 -> 0.9851), group features (+0.0025 held-out; +0.0023 public LB for the sibling features alone) and cross-encoders on close calls (+0.0070 held-out); for France, without labels, rules had to be checked with a data-generator artifact and France-only leaderboard probes because rules that hold in US/India labels failed there.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` holds all source in `src/`, the France analysis scripts in `france_fix/`, the unattended run scripts of 27 Sep in `overnight/`, the lab-server drivers in `server/`, `README.md` and `requirements.txt`.

- Entry points, in order (full commands, outputs, run times and machines in `README.md`): `learn_maps.py` -> `prep.py` -> `embed.py train|encode` -> `pipeline.py build train full` -> `pipeline.py train full` (stage 1) -> `pipeline.py train full cons` (stage 2) -> `pipeline.py build test test` -> `pipeline.py predict full_cons test` -> `rerank.py select` -> `topk5.py` -> `rerank.py extras` -> `rerank.py train|score|stage3` (France transformers) -> `blend.py` + `fr_minrule.py` (France scores) -> `ce_folds.py` (3-fold e5-small; 3-fold bge on a 48 GB GPU) -> `blocking_b2.py build|predict|rows` (wide US/India test search) -> `rerank.py score|stage3` -> `blend.py` -> `fr_minrule.py` -> `finalize.py` -> France rule sets + `finalize.py` -> `server/run_v9.sh` (second generator: the same pipeline with `BER_V8=1`, lab server) -> `assemble_final.py` -> `make_candidates.py`.
- `output/matching_results.tsv` is written by `assemble_final.py`, `output/candidate_pairs.tsv` by `make_candidates.py`. Rebuilt from the saved intermediate files on 27 Sep 2026 (README steps 16, 18 and 19), both matched the submitted files byte for byte, and the official validator (with `--check-ids`) printed PASS.
- Hardware: laptop (Intel i5-12450H, 16 GB RAM, RTX 3050 Laptop 4 GB) for all steps except the bge-reranker-v2-m3 folds (friend's RTX A6000 48 GB) and the second generator's run (lab server, RTX PRO 6000 Blackwell 96 GB). About 27 h of laptop compute (sum of the step times in `README.md`), 10-12 h on the A6000.

### B. Additional Results

**Cross-encoders on the same held-out close calls** (AUC; fold k = labelled close calls whose record is in fold k, records whose candidates span both S1 halves left out, no compared model trained on them):

| Fold (pairs) | e5-small halves models (v7ens) | e5-small, 3 folds | e5-base, 3 folds (800k training rows) | bge-reranker-v2-m3, 3 folds |
|---|---|---|---|---|
| 0 (557k) | 0.98875 | 0.99121 | 0.99007 | 0.99392 |
| 1 (551k) | 0.98891 | 0.99144 | 0.99081 | 0.99412 |
| 2 (551k) | 0.98830 | 0.99084 | 0.98994 | 0.99373 |

The bge column was computed from the saved out-of-fold score files on the same pairs. On every held-out row of each fold (837k-846k pairs per fold, including the stage-1 ranks 3-5 and records whose candidates span both S1 halves) bge-reranker-v2-m3 against e5-small is 0.99022 vs 0.98723 (fold 0), 0.99034 vs 0.98731 (fold 1), 0.98995 vs 0.98676 (fold 2).

Stage 3 per family, held-out macro F0.5 (all close calls rescored): e5-small 0.98903, bge 0.98951, blend 0.98950. The e5-base family (0.98890) was not used in the final file.

**Transformer as the final matcher** (sample of 80,218 train S1 rows with all their records, 15.07M candidate pairs, same top-5 candidates for every method; optimistic because the sample has fewer look-alikes, so only comparisons within the table hold): GBDT (stage 1 only, no group features) 0.98998; fine-tuned bge-reranker-v2-m3 alone 0.99321; stack of both 0.99362; stack on close calls only (34% of records) 0.99351. This is why the transformers score close calls rather than every pair.

**Blocking audit** (100k labelled Latin-name records per country against all train S1 rows of the country; recall / candidates per record):

| Step | US | India |
|---|---|---|
| production shortlist | 0.9868 / 31.4 | 0.9784 / 30.3 |
| + name 3-gram cap 4,000 -> 20,000 | 0.9910 | 0.9836 |
| + word searches keep every term | 0.9921 | 0.9852 |
| + no-address name top 30 -> 100 | 0.9943 / 34.5 | 0.9884 / 33.3 |
| + fine-tuned e5 top 10 for every record | 0.9971 / 42.0 | 0.9956 / 41.4 |

At the production cap of 4,000, a median US name kept only 4 character 3-grams in the search and 3.2% kept none; exact-name records were found 86% of the time.

**Tried and not used:**
- Training on resampled "test-like" data (S1 rows removed by deleting their pairs): public LB -0.0086 (v4, 0.961357). Deleting pairs halved the candidate lists (15.9 per US record vs 31.3 on test), which the real search at lower density does not do.
- French-specific cleaning (dotted legal forms, street abbreviations, "et" for "&"): public LB fell (v6, 0.967332 vs 0.9700); v6 also added the consensus features, which gained +0.0015 on a test-density check, so the drop was attributed to the French cleaning; not used.
- Self-training on test pseudo-labels: held-out 0.97921 vs 0.97926 without (test-like validation).
- Isotonic calibration with per-country rules: +0.0002 held-out, left out because it added ~56k borderline test matches.
- One joint stage 3 over both transformer families: +0.00002 held-out (noise level).

**Checks for shortcuts and leakage:** Spearman correlation between S1 id number and matched id number 0.0003; between file row positions -0.002; test records with a train S1 at cosine >= 0.9 while their best test S1 is below 0.6: 0.00%. No id, order or train/test overlap signal is used.

**Models and licences** (parameters counted from the weight files): intfloat/multilingual-e5-small, MIT, 117.7M; intfloat/multilingual-e5-base, MIT, 278.0M; BAAI/bge-reranker-v2-m3, Apache-2.0, 567.8M; LightGBM (MIT) and XGBoost (Apache-2.0) tree ensembles. The largest model has under 0.6B parameters (limit 8B). The second generator's run uses bge-reranker-v2-m3 as well (named in `server/run_v9.sh`). No external data, APIs, gazetteers or lookup services were used; the normalisation tables and the tree models were fitted on the provided training data, the transformers are the public checkpoints above fine-tuned on the provided training data, and test data was used only for unsupervised statistics and the leaderboard submissions described above.
