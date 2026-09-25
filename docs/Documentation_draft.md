# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Team name as registered]
**Team Members:** [A], [B], [C], [D]
**Submission Date:** 27 Sep 2026

> DRAFT — numbers marked (S10) come from a 10% validation slice and will be replaced by the full-train numbers.

---

## 1. Executive Summary
We treat the task as **assigning every Source 2 / Source 3 record to at most one Source 1 entity** (the training labels never link a record to two S1 entities). A cheap, recall-oriented shortlist (character-3-gram TF-IDF on names and addresses, plus a fine-tuned multilingual embedding search for names written in Indian scripts) is scored by a two-stage LightGBM matcher. A final set-selection step chooses each S1 entity's match list to maximise expected F0.5. Validation macro F0.5 is 0.9879 (S10), up from 0.8926 for a hand-written rule.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training files (2.2M S1, 5.0M S2, 5.3M S3 rows):
- Only **5.6%** of S1 entities are singletons; the average entity has **3.46** matches (1–3 from each of S2 and S3).
- **No S2/S3 record matches two S1 entities** (0 of 7.6M), and **matched records always share the country label**.
- **26%** of S2/S3 records match nothing (decoys).
- **39%** of S1 rows share their exact name with another S1 row (chains such as "Primary Care Group" ×253), so the name alone cannot decide.
- How matched names differ: identical after lowercasing 25.5%; only legal form / title differs 25.2%; words dropped/added 11.7%; typos 12.8%; web-domain form ("exultherbal.com") 4–8%; **non-Latin script 7.2%** (Hindi, Bengali, Odia, Tamil, Telugu, Kannada, Malayalam, Gujarati); invented alias names ~4% (only the address links them).
- How matched addresses differ: short forms/typos 52.7%; state written in a local script 9.2%; parts missing 7.4%; house number changed or truncated 6.8%; missing entirely 4.4%.
- Test adds **France** (15% of test S1) with no training labels, so every feature is country-agnostic.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier + expected-F0.5 set selection.
**Core Innovation:** (1) record-to-entity assignment that exploits the one-S1-per-record structure; (2) a multilingual embedding model fine-tuned on training pairs that lifts cross-script name search from 15.6% to 99.8% recall@10; (3) normalisation tables learned from the training pairs rather than written by hand.

---

## 3. Candidate Generation (Blocking)
Searched **from each S2/S3 record** to S1 records **of the same country** (the union of):
- **Name:** TF-IDF on character 3-grams of the cleaned, space-free core name (legal forms and titles removed, web domains split back into words, "d/b/a" aliases dropped), top 10 by cosine; top 30 for records with no address.
- **Address:** TF-IDF on character 3-grams of the cleaned address (learned short forms such as rd→road, local-script state names → English), top 10.
- **Embedding (non-Latin names only):** `intfloat/multilingual-e5-small` (MIT, 118M parameters) fine-tuned with in-batch contrastive loss on 276k (non-Latin name + address, S1 name + address) training pairs; top 10 by cosine on GPU.
- To keep search cost bounded as S1 grows, 3-grams shared by more than 4,000 (name) / 2,000 (address) S1 rows are ignored during search only.

- **Blocking keys used:** character-3-gram TF-IDF on name and address, fine-tuned multilingual embeddings, same-country constraint.
- **Candidate pairs generated:** ~19.7 per S2/S3 record (S10). [test total: TBD]
- **How you ensured true matches were not lost:** three complementary searches measured separately on training data; recall ceiling **99.61%** (S10), up from 98.69% with name+address TF-IDF only. The embedding search removed 60% of the remaining misses (the non-Latin-script ones).

---

## 4. Matching Model

**Features used (52):**
- Name: TF-IDF cosine (full vocabulary), Levenshtein ratio, token-set / token-sort / partial ratio, Jaro-Winkler, ratio with legal forms kept, exact-core-name flag, embedding cosine.
- Address: TF-IDF cosine, ratio, token-set, partial ratio; house numbers (count, common count, first-number equal / contained / prefix, relative difference, digit edit distance); alphanumeric unit tokens such as "8-9-1/14a" (overlap).
- Competition among a record's candidates: margin to the best other candidate and rank for name cosine, address cosine, token-set scores and embedding cosine; number of candidates; number of near-identical names / addresses.
- Other: chain-name count (S1 rows sharing the core name), source (S2/S3), flags for web-domain name, alias, non-Latin script, missing address.
- **Stage 2:** stage-1 probability, its margin to the record's 2nd candidate, and S1-side competition: how many other records already point to this S1 with high probability (overall and from the same source), their best probability, and this record's rank among them.

**Model type:** LightGBM (stage 1: 255 leaves, 600 rounds, ~8M sampled pairs; stage 2: 127 leaves, 500 rounds on each record's top-2 candidates).
**Threshold selection method:** each record goes to its highest-probability S1; per S1, the kept set maximises expected F0.5 = 1.25·Σp(top k) / (k + 0.25·Σp(all)), compared with the probability of no match. Chosen over a single tuned threshold because it scored higher on out-of-fold predictions (0.98794 vs 0.98770, S10).

Validation: 3 folds grouped by true S1 entity (all records of one business in the same fold). Scores are computed only on S1 entities whose matches were **not** used to fine-tune the embedding model, and the matchers are trained only on those records, so the embedding feature cannot leak.

---

## 5. Results & Error Analysis

| Step | Recall ceiling | Macro F0.5 (OOF) |
|---|---|---|
| Rule: best name+address token-set similarity, threshold | 0.987 | 0.8926 |
| LightGBM, 40 features (v1) | 0.987 | 0.9771 |
| + embedding search, full-vocab cosines, margin/rank features, house-number tokens | 0.996 | 0.9871 |
| + stage 2 (S1-side competition) | 0.996 | 0.9877 |
| + expected-F0.5 set selection | 0.996 | 0.9879 |
| [full training data] | TBD | TBD |

- **F_0.5 Score (macro):** [final validation score]
- **Common false positives (wrong merges):** near-duplicate records of a different business at almost the same address ("Consolidated Médical Studios Ltd, 31 Laurel Circle" vs S1 "... LLC, 315 Laurel Circle"); unrelated names at an identical address.
- **Common false negatives (missed matches):** records with **no address** whose name exists at several S1 addresses (chains) — 73% of wrong-entity errors; heavily re-worded names with partial addresses.
- **Concrete fix example:** "Royal Enterprises Private Limited" written as "रॉयल एंटरप्राइजेज प्राइवेट लिमिटेड": TF-IDF cannot match across scripts, so in v1 such records were 60% of all shortlist misses; the fine-tuned embedding search reduced that to 0.2%.

---

## 6. Conclusion
Exploiting the data's structure (one entity per record, same country) and measuring each error type before fixing it gave most of the gain; the learned cross-script search fixed the largest error class. Remaining errors are dominated by address-less records of chain businesses, which the data cannot disambiguate.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `learn_maps.py` → `prep.py` → `embed.py train|encode` → `pipeline.py build train full` → `pipeline.py train full` → `pipeline.py build test test` → `pipeline.py predict full test`. See `README.md`.

### B. Additional Results
- Models and licences: multilingual-e5-small (MIT, 118M); LightGBM (MIT). No external data or lookups.
- Hardware: one laptop (RTX 3050 4 GB, 16 GB RAM). Runtime: [TBD].
