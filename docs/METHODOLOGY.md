# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Mommy's Good Boys  
**Team Members:** Kanav Bhardwaj, Bhavya Jain, Harsh Gupta, Gathik Jindal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Every Source 2 / Source 3 record is searched only against the Source 1 (S1) rows of its own country, with four bounded top-k searches (character and word TF-IDF, and a fine-tuned multilingual embedding). A cascade then decides: a LightGBM model keeps each record's best 1-2 candidates, an XGBoost model adds features that compare all records pointing at the same S1 row, and fine-tuned cross-encoders (transformers that read the two records' text together) re-score the 23.5% of test records whose decision is uncertain. A second run of the same code with wider blocking gives a second probability per pair, and a LightGBM stacker combines both runs for US and India. France (15% of the test set) has no training labels; its rules were checked with label-free tests built on artefacts of the data generator, with the analogous pairs in labelled US/India data, and, for some rules, with uploads that changed only France rows. An expected-F0.5 rule then chooses each S1 row's match set.

| Result | Value |
|---|---|
| Public leaderboard, final upload (version v10d) | **0.990565** |
| Held-out macro F0.5, US/India (half of the train S1 rows, searched against all train S1 rows of the country) | **0.99226** |
| `candidate_pairs.tsv` | **12,249,116 pairs = 7.07 per S1 row**; reduction ratio 0.99999818 against all same-country pairs |
| Share of held-out true pairs inside the same candidate definition (recall ceiling) | 0.98936 |
| Largest model | 568M parameters (bge-reranker-v2-m3, Apache-2.0). No external data, APIs or lookups |

The sources of these numbers are given in sections 3 and 5.

---

## 2. Methodology

**Terms used in this document**

| Term | Meaning |
|---|---|
| S1 row | one Source 1 entity (the deduplicated reference source) |
| record | one Source 2 or Source 3 row. Every search starts from a record and looks for S1 rows. |
| pair | one (S1 row, record) combination |
| decoy | a record that matches no S1 row. Decoys are 26% of train records and are made to resemble an S1 row. |
| macro F0.5 | F0.5 per S1 row, averaged over all S1 rows; an S1 row with no true match scores 1 only for an empty prediction (the competition metric) |
| held-out half | train S1 rows with `zlib.crc32(s1_id) % 1000 < 500` (1,102,482 rows) and their records. The embedding model was fine-tuned on the other half only. |
| full density | the search index holds all train S1 rows of the country, as on test. Every held-out score here is at full density. |
| out-of-fold | a score from a model that did not train on that row (3 folds, grouped by true S1 row, so all records of one business share a fold) |
| p1, p2, p3 | probability of a pair from stage 1, stage 2 and stage 3 |
| close call | a record whose best p2 is in [0.01, 0.995], or whose 2nd candidate has p1 >= 0.2 |
| family | one cross-encoder model type trained as 3 fold models: multilingual-e5-small, bge-reranker-v2-m3 and multilingual-e5-large in the final file (a multilingual-e5-base family was used in v7g only); France rows use the half models A, B, A2 instead |
| A, B, A2 | France half models: e5-small A trained on the close calls of the non-held-out half, e5-small B on those of the held-out half, e5-base A2 on 420k of A's rows |
| singleton | an S1 row with no true match (5.58% of train S1 rows) |
| AUC | area under the ROC curve: the chance that a random true pair scores above a random false pair |
| log loss | mean negative log-probability given to the true label |
| split gain | the loss reduction LightGBM credits to a feature over all its splits |
| "halves" and "all rescored" | two held-out protocols. "Halves" (versions v7 to v7ens): records whose candidates fall in both S1 halves keep their GBDT probability. "All rescored" (3-fold designs): every close call has an out-of-fold transformer score. Scores from different protocols are not compared. |
| second pipeline | the same code run with wider blocking (`BER_V8=1`) by team member Gathik Jindal on a lab GPU server, with its own models |
| GBDT | gradient-boosted decision trees (LightGBM or XGBoost) |
| public LB | public leaderboard score |

**Where the numbers come from.** Each number carries its source in brackets:
- [E: *name*]: the section of that name in `EXPERIMENTS.md`, the team's lab notebook;
- [L]: `submissions/LOG.md`, the log of every built and uploaded version; [F: *version*]: `submissions/<version>/finalize.json`;
- [R: *file*]: a README or a script docstring in `code/business_entity_resolution/`;
- [M]: measured for this document on 27 Sep 2026 by the scripts in `src/experiments/measure/` (named next to each number), from the saved files (`work/` intermediate files, `code/business_entity_resolution/sets/`, `submissions/v10d/`); [M: *file*] means the number was read from that file.

In the zip, copies of `EXPERIMENTS.md`, `submissions/LOG.md`, the `finalize.json` files of v9z-v10d and the runbooks are in `code/business_entity_resolution/docs/`. The `work/*.log` run logs stay on the laptop; every number used is copied here. Estimates are marked "estimate".

### 2.1 Problem Analysis

**Size** [M: `data_facts.py`]

| | S1 rows | Records | Records per S1 row |
|---|---:|---:|---:|
| Train, US | 1,323,633 | 6,186,873 | 4.67 |
| Train, India | 883,188 | 4,133,346 | 4.68 |
| Test, US | 663,106 | 3,817,031 | 5.76 |
| Test, India | 809,986 | 4,717,565 | 5.82 |
| Test, France | 259,452 | 1,434,993 | 5.53 |
| Test, all | 1,732,544 | 9,969,589 | 5.75 |

**Label structure** [M: `data_facts.py`, from `train_ground_truth.tsv`]
- The 7,638,365 true pairs involve 7,638,365 distinct records, so each matched record belongs to exactly one S1 row.
- Every true pair has the S1 row and the record in the same country, so every search runs within one country.
- 74.0% of train records are matched; 26.0% are decoys.
- 5.58% of S1 rows have no match. The mean is 3.46 matches per S1 row.
- **Chains.** 38.3% of S1 rows share their exact name with another S1 row of the same country; 50.0% after cleaning names to the core name (legal form, titles and punctuation removed).
  - The largest groups are US "Primary Care Group" (253 rows), "Ear Nose & Throat Group" (251) and "Pediatric Group" (222).
  - For these, the name alone cannot pick the S1 row.

**Noise in true pairs.** Measured [M: `change_sample.py`, `change_table.py`] with the change detector `src/france_fix/artifacts/census/ops.py` (rules that recognise each kind of change between a record and an S1 row) on a random sample of 80,000 true pairs per country. One pair can carry several changes, so columns do not sum to 100%.

| Name change | US | India |
|---|---:|---:|
| no change detected | 8.4% | 5.7% |
| case changed (upper, lower, title case) | 17.9% | 11.2% |
| legal form dropped / added | 14.7% / 7.9% | 8.5% / 3.4% |
| legal form changed / abbreviated (Private -> Pvt) / dotted (S.A.S.) | 0.7% / 1.1% / 3.3% | 11.7% / 7.5% / 1.5% |
| one word swapped for another | 14.1% | 11.5% |
| word dropped / word added | 8.3% / 4.1% | 3.1% / 6.4% |
| typo | 6.1% | 4.5% |
| letters replaced by digits ("Seaf0od") | 2.7% | 2.1% |
| accents added or removed | 7.2% | 5.3% |
| written as a web domain ("unitedfoundation.com") | 5.6% | 4.6% |
| written in an Indian script | 0.0% | 18.0% |
| word order changed | 4.3% | 1.2% |
| title added or dropped (Mr, Smt, Sri) | 1.4% | 6.5% |
| "trading as" alias / "&" swapped with "and" / acronym | 0.9% / 0.9% / 0.1% | 0.7% / 0.5% / 0.3% |

| Address change | US | India |
|---|---:|---:|
| address missing | 4.7% | 4.0% |
| written in upper case | 42.9% | 11.5% |
| street type shortened (Road -> Rd) | 44.2% | 0.0% |
| state written in full (OH -> Ohio) / state abbreviated | 42.2% / 0.0% | 0.0% / 33.4% |
| written in an Indian script | 0.0% | 22.7% |
| components reordered | 25.8% | 31.7% |
| a component dropped / added / changed | 9.7% / 3.6% / 23.7% | 57.0% / 6.6% / 10.3% |
| typo in a component | 8.5% | 13.6% |
| unit (suite, floor) dropped | 10.7% | 6.0% |
| house number missing | 8.2% | 1.6% |
| first number moved up by 1-20 / down by 1-20 | 0.8% / 0.8% | 1.6% / 1.9% |

**Test differs from train** [E: Test-side EDA; LB gap analysis; Shortcut checks]
- Test has 5.5-5.8 records per S1 row against 4.67-4.68 in train: more decoys per S1 row.
- The uncertain band (best-candidate probability 0.05-0.5) holds 3.0-3.8% of records in train (out-of-fold) and 7.8-8.9% on test.
- Most uncertain test records are the same name on the same street with the house number shifted a few units ("9692 Diamond Rd" vs S1 "9687 Diamond Rd"). In train, such pairs are 92.7% false.
- Records whose best candidate is such a shifted-number look-alike: US 14.3% of train records vs 21.9% of test records; India 2.2% vs 3.2%.
- US test S1 is half as dense as US train S1 (663,106 vs 1,323,633 rows). The mean US chain size (S1 rows sharing the core name) of best candidates is 22.0 in train and 11.7 in test.
- Confident pairs (p1 >= 0.9) look the same in train and test (US name / address token-set similarity 93.9 / 94.9 vs 93.9 / 94.5). So true matches on test are not noisier; the extra difficulty is the decoys.
- Records without an address: train US 3.6% / India 3.0%; test US 2.9% / India 2.4% / France 3.0%.

**France: 15.0% of test S1 rows, no labels** [E: v5 result and France diagnosis]
- Because every S1 row counts equally, the leaderboard is about 0.85 x US/India + 0.15 x France.
- France S1 has "twins": the same generic name ("<city> <word> SARL") on the same street, differing only in legal form (SA / SAS / SARL / SASU), a descriptor word or the house number.
  - Records whose 2nd candidate also has p1 > 0.5: France 2.07%, US 0.39%.
- The test cities are Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac and others; there is no Paris, Lyon or Marseille. 5-digit postcodes appear in 0.4% of France S1 addresses.
- France is not under- or over-matched: v3 predicted 3.43 matches per S1 row for France, 3.38 for US and 3.34 for India, against 3.46 in train truth. Its loss comes from confident choices of the wrong record.

**Artefacts of the data generator** [M: `data_facts.py` for artefact 1; `change_sample.py`, `lowercase_test.py` for artefact 2]. These two regularities, measured on labelled US/India data, became the label-free tests used for France (section 4).

*Artefact 1: decoys move the house number up; true records do not.*
- Method: first number of the record minus first number of the S1 row.
- True pairs use their true S1 row. Decoys use their best stage-1 candidate (out-of-fold table `work/models/full_cons/oof.parquet`).
- Rows not shown: no number on one side, or a difference beyond +-20.

| Difference | US decoys (1,608,336) | US true pairs (4,578,522) | India decoys (1,073,496) | India true pairs (3,059,843) |
|---|---:|---:|---:|---:|
| 0 | 5.4% | 77.0% | 21.1% | 65.8% |
| +1 | 7.5% | 0.23% | 4.1% | 0.45% |
| +2 to +5 | 30.1% | 0.29% | 15.2% | 0.77% |
| +6 to +20 | 30.4% | 0.27% | 15.9% | 0.69% |
| -1 to -20 | 0.57% | 0.81% | 3.3% | 2.1% |

*Artefact 2: true records whose name adds or swaps in a noise or descriptor word are almost never all-lowercase; decoys with the same change are all-lowercase at the ordinary rate (2.4-3.0%).*
- Method: the change detector on a random sample of 80,000 true pairs and 80,000 decoys (with their best candidate) per country.
- Rows: records whose name adds or swaps in a noise word (group, services, associates, fils, cie ...) or a descriptor word (club, ecole, centre, amicale, association ...), from the word lists in `ops.py`.

| Records with a noise/descriptor word added or swapped in | All-lowercase share |
|---|---:|
| US true (6,799) | 0.03% |
| India true (6,753) | 0.22% |
| US decoys (23,643) | 3.01% |
| India decoys (23,220) | 2.39% |

- The test does not work for other words: records with a changed word outside those lists are all-lowercase 2.84% (US true) vs 2.81% (US decoys).
- So the all-lowercase share of a France pair set of this kind estimates its true share: near 0.2% means mostly true, near 2.4-3.0% means mostly decoys.

**No shortcuts** [E: Shortcut checks; v5 result and France diagnosis]
- The distance between S1 id and record id is distributed the same for true and random pairs (share below 10M: 1.99% vs 1.99%).
- File row order carries no signal: Spearman correlation -0.002 / -0.001.
- Test records are not copies of train businesses. Test records whose best train S1 has cosine >= 0.9 while their best test S1 is below 0.6: 0.00%.

**Validation must use the full index** [E: FULL setup].
- A first validation on a 10% slice of train S1 rows hid a blocking failure.
- The search caps on document frequency were tuned on the slice. At full density they dropped every common city and street 3-gram.
- Held-out recall of the candidate list fell to 0.7968 (US 0.742, India 0.879). Test has full density, so the same failure would have hit test.
- Since then every blocking and model number is measured with all train S1 rows of the country in the index.

### 2.2 Solution Strategy

**Approach Type:** Blocking + cascade of classifiers (hybrid: gradient-boosted trees and fine-tuned cross-encoder transformers). Two pipelines with different blocking are combined per pair, and an expected-F0.5 rule selects each S1 row's match set. France uses label-free rules on top of the same models.

**Core Innovation:**
1. **Validation at full density** (every S1 row of the country in the index, as on test). The held-out half is searched against all train S1 rows of its country, and every blocking choice was re-measured this way (recall 0.7968 -> 0.9851, section 3).
2. **Group features.** Stage 2 compares a record with the other records that point at the same S1 row. A true record agrees with its siblings on address and house number; a decoy does not. This gave +0.0025 held-out and, for the sibling features alone, +0.0023 public LB (section 5).
3. **Transformers only where the trees are unsure.** Cross-encoders score only the 23.5% of test records that are close calls. Each family is trained as 3 fold models, so every labelled close call has an out-of-fold score and a stage-3 model can learn how far to trust it. The first cross-encoder gave +0.0070 held-out (0.98094 -> 0.98794).
4. **Two blocking designs, combined per pair.**
   - The second pipeline's wider blocking adds true pairs our lists miss: the recall ceiling rises from 0.98042 to 0.98936 at 5.35-5.68 candidates per held-out S1 row.
   - Its probability is blended with ours (v10a), then a stacker learns the combination (v10d).
5. **France without labels.** Each France rule used one or more of three checks:
   - the analogous pairs in labelled US/India data, which can mislead: descriptor swaps are 98.9% true in US but decoys in France;
   - a label-free test built on a generator artefact (lowercase share, house-number direction);
   - uploads that changed only France rows (v7i, v7j, v7m, v10b).

   Several rules reached the leaderboard only inside bundled uploads (v9y, v9zm, v10d).

**Validation protocol** (used for every held-out number unless marked otherwise)
- Train S1 rows are split into halves by `crc32(s1_id)`. Scores are reported on the held-out half (1,102,482 S1 rows, 3,815,794 true pairs).
- Candidate search for held-out records runs against all train S1 rows of the country.
- Stages 1, 2 and 3 and the 3-fold cross-encoders are trained in 3 folds grouped by true S1 row and scored out-of-fold. The older France half models (A, B, A2) and the stacker are cross-fitted by S1 halves instead (section 4).
- The score is macro F0.5 over the held-out S1 rows, singletons included, computed as the competition defines it.
- US and India only: France has no labels.

**Pipeline at a glance** (test pair counts [M: `test_candidates.py`])

```
record (S2/S3) -> cleaning (tables learned from train matches only)
  -> 4 top-k searches against the S1 rows of the same country        raw search output: 307.2M pairs
                                                                     (wide US/India search: 469.8M)
  -> stage 1  LightGBM, 55 pair features: keep each record's top 1-2 stage-2 input: 10.7M pairs, 6.17 per S1 row
  -> stage 2  XGBoost, +21 group features                            -> p2
  -> close calls (23.5% of records) + their stage-1 ranks 3-5        3.79M pairs
       -> cross-encoders, 3 fold models per family (e5-small, bge-reranker-v2-m3, e5-large)
       -> stage 3  XGBoost per family                                -> p3
  second pipeline: same code, wider blocking, own models             -> its own p per pair
  US/India: blend of both pipelines (v10a), then a LightGBM stacker over both (v10d)
  France:   older transformer set; transformers may only lower p; label-free rules and pair sets
  -> expected-F0.5 set per S1 row                                    -> matching_results.tsv
  every pair the final models scored                                 -> candidate_pairs.tsv (12.25M, 7.07 per S1 row)
```

**Build order** (each version is one row of `submissions/LOG.md`; results in section 5)
- v2 to v7ens (25-26 Sep): blocking, the GBDT cascade, cross-encoders on close calls.
- v9y (27 Sep): 3-fold cross-encoders, wide US/India search, second-pipeline pairs, three France rules.
- v9zm: France recoveries and a US/India fix where the two candidate builds disagree.
- v10a: US/India blend with the second pipeline.
- v10b / v10c: France matches from the second pipeline and three verified France pair sets.
- v10d: US/India stacker and one France veto. Final upload.

---

## 3. Candidate Generation (Blocking)

#### How the search works

- Every search runs **from each record towards the S1 rows of the same country** and returns a fixed number of candidates (top-k). No pair is ever compared exhaustively.
- A search is a sparse matrix product with a per-row top-k (`sparse_dot_topn`), run in chunks of 100k-250k records against one country's S1 index.
- Terms that occur in more than a fixed number of S1 rows (the document-frequency cap, `max_df`) are skipped during the search. This bounds the work per record as S1 grows.
  - The similarity features of section 4 use the full vocabulary for the character 3-grams. The two word-level cosines reuse the capped search vectors.
- Cleaning before the search:
  - accents stripped;
  - legal forms and titles removed from the "core name";
  - web domains split back into words;
  - "d/b/a" aliases dropped;
  - address short forms (rd -> road) and local-script state names mapped with tables learned from train matches (`learn_maps.py`).

#### The four searches, and why each exists

| Search | Key | Top-k | Term cap | Why it exists | Recall alone, full density [E: Blocking at full density] |
|---|---|---|---|---|---|
| Name | TF-IDF of character 3-grams of the space-free core name | 10; 30 for records without an address | 4,000 S1 names | typos, squashed names, web domains, records whose address is missing | 0.555 (US, 50k held-out records) |
| Address | TF-IDF of word 1-2-grams of the cleaned address | 10 | 5,000 S1 rows | records whose name was replaced (invented or web-domain names) | 0.894 |
| Name + address | TF-IDF of word 1-2-grams of core name + address | 20 (wide US/India test search: 40) | 5,000 (wide: 20,000) | strongest single search; separates branches of a chain | 0.982 |
| Embedding | multilingual-e5-small fine-tuned on (non-Latin record, S1) train pairs of the other half, text "name \| address", 64 tokens; exact cosine top-k on GPU | 10 | none | names written in Indian scripts share no characters with the Latin S1 name | recall@10 0.998 against all 883k India S1 rows [E: Embedding search] |

- Union of the first three at full density: US 0.9856, India 0.9593 (India before the embedding search). Search time about 14 s per 50k records [E: Blocking at full density].
- Embedding search, recall@10 on 20k held-out non-Latin records: 0.156 off the shelf, 0.573 fine-tuned on names only, 0.998 fine-tuned on name + address [E: Embedding search].
- Held-out recall of the whole union (held-out half, full density): **0.9851**. The first version, with a character 3-gram address search capped at 2,000, had 0.7968 [E: Run 9].

#### Learned filter (stage 1)

- The stage-1 LightGBM model (section 4) scores every searched pair.
- Per record, it keeps the best candidate, plus the second one when its p1 >= 0.01. This is the stage-2 input.
- For close-call records it also keeps stage-1 candidates ranked 3 to 5 with p1 >= 0.005. The cross-encoders and stage 3 score these too ("ranks 3-5").

#### Wide search for US/India test records

`blocking_b2.py`: the name + address search with cap 20,000 and top 40, instead of 5,000 and top 20.
- Why [R: `blocking_b2.py` docstring]: most with-address misses sit at busy addresses, where the top-20 search runs out of slots or drops common words. On labelled train samples this search finds 45% of those misses, and stage 1 ranks 94% of them first. Estimated held-out gain +0.0011 to +0.0015 (estimate; the full wide build exists only for test).
- On test, 13.8k of the US/India pairs it added after stage 3 have an S1 row that the production lists never contained [L: v7p].

#### Second pipeline's blocking (audited)

A blocking audit on 100k labelled records per country found four fixes (table in Appendix B.3) [E: Blocking and cleaning audit]:
- name search cap 4,000 -> 10,000 or 20,000;
- word searches keep terms that occur once (min_df 1);
- name top 100 for records without an address;
- the fine-tuned e5 top-10 search for every record.

Two cleaning fixes came with them:
- **India state codes.** The pooled map sent `tn` to "tennessee".
- **Spelled-out ordinals.** "eleventh street" is mapped to 11.

The second pipeline runs our code with these settings (`BER_V8=1`, name cap 10,000):
- its search output keeps 0.9963 of held-out true pairs at 40.5 candidates per record, against 0.9851 at 30.9 for ours [E: v9 GBDT results; R: code README step 3];
- the audit is worth +0.0032 held-out before any transformer (GBDT 0.98151 -> 0.98472).

#### Third source: the same-address rule (countries without training labels)

`same_address.py` proposes pairs for the countries without training labels (France) that neither search finds [R: `same_address.py` docstring; code README step 26]. It looks at the records that the matching file (v10b) leaves unmatched, and compares each one only with the S1 rows at exactly the same address.
- **Why it exists.** The data generator rewrites some true copies as an acronym ("AC" for "Animation Culture SCI") or a web domain. Such a name shares almost no text with the S1 name, so the name search misses it. At an address with several businesses, the address searches' top-k lists can also leave the S1 row out. 141 of the 223 candidate pairs this rule adds are in no score table of either pipeline (table below).
- **Blocking key.** The key is the house number(s) plus the set of street words of the first address component that holds a number.
  - Accents and filler words (de, du, la, of, the ...) are dropped, and street types are shortened (rue -> r, avenue -> ave).
  - A standalone 5-digit number is read as a postcode and ignored. When the component has no street word, the next component is added.
  - The postcode reading fits France (0.4% of France test S1 addresses hold a standalone 5-digit number) but would drop 5-digit house numbers, which 10.9% of US test S1 addresses hold (both shares counted on 28 Sep 2026 in `test_source1.tsv`). The rule runs only on the countries without training labels.
  - Records and S1 rows of the same country with the same key form the candidate pairs. Keys shared by more than 50 S1 rows are skipped.
- **When a pair is accepted.** All of these must hold:
  - the change detector `ops.name_ops` (also used by the second-pipeline recall of section 4) finds only generator noise between the two names: acronym or initials, web domain, squashed words, word order, digits for letters, case, accents, brackets, hyphens, commas, '&' for 'and', a legal form dropped, dotted or abbreviated;
  - exactly one S1 row at the address passes that test;
  - the pair is not in one of the France veto sets of section 4, and the second pipeline does not match the record to another S1 row;
  - an acronym name needs a single S1 row at the address;
  - any other name needs no same-name S1 row on the same street at another number and no changed place or street component, and a web-domain name must spell the S1 name exactly;
  - the S1 row already has a match: the rule never fills an empty S1 row.

| Stage (test, France) [R: `same_address.py` docstring; the same counts are printed by `same_address.py` in every `build_final.py` run, e.g. `src/logs/build_final_28sep.log`] | Pairs |
|---|---:|
| unmatched France records (with an address key) | 577,492 (538,129) |
| same-address pairs that pass a cheap name prefilter | 42,511 |
| names that differ only by generator noise | 849 |
| the only such S1 row of the record | 709 |
| not vetoed, not matched elsewhere by the second pipeline | 693 |
| safety filter | 309 |
| S1 row already matched: `sets/fr_same_address_safe.parquet` | **298** (239 acronym, 59 other names) |

- **In the final files.** v10c adds the 298 pairs as matches, and v10d still matches all 298 [M: `candidate_sources.py`, run on 28 Sep 2026].
  - 66 were already in the scored candidate list, and 9 in the second pipeline's matching file.
  - The other **223** are new candidate pairs. They form the same-address row of the table below; 217 of them are acronym names.
- **Labelled analog.** On the held-out half of train, the rule up to the one-S1-row test gave 651 pairs for records our held-out prediction leaves unmatched, and 99.85% of them are true pairs (US 509, India 142) [R: `same_address.py` docstring].
- **Reproduction.** `build_final.py` regenerates the 298 pairs from the cleaned test files and the rebuilt v10b. It stops unless they equal the shipped set, then applies the shipped set, so both output files keep their md5 (Appendix A).

#### What `candidate_pairs.tsv` contains

The problem statement defines the candidate file as the pairs the final model runs inference over. Ours is the union of the pairs that a final model scored, plus the matched pairs that the France steps took from the second pipeline and from the same-address rule (the third source, above) [M: `test_candidates.py`, `candidate_sources.py`; the last two rows from the log of `make_candidates.py`]:

| Part | US | India | France | All |
|---|---:|---:|---:|---:|
| Stage-2 input (wide search for US/India, production search for France) | 3,942,489 | 5,012,866 | 1,732,369 | 10,687,724 |
| + stage-1 ranks 3-5 of close-call records | 224,156 | 401,696 | 328,297 | 954,149 |
| = our final score table (`test_scores_blend_v7p.parquet`) | 4,166,645 | 5,414,562 | 2,060,666 | 11,641,873 |
| + second pipeline's US/India stage-2 pairs not already present (the v10d stacker scores them) | 178,789 | 422,862 | 0 | 601,651 |
| + pairs restored by the US/India list-mover fix (section 4) | 4 | 25 | 0 | 29 |
| + France pairs of the second pipeline's matching file that the final file matches, not already present | 0 | 0 | 5,340 | 5,340 |
| + pairs of the France same-address rule (`same_address.py`, above) that the final file matches, not already present (141 of them are in no score table of either pipeline) | 0 | 0 | 223 | 223 |
| **= `candidate_pairs.tsv`** | **4,345,438** | **5,837,449** | **2,066,229** | **12,249,116** |

- Every matched pair of `matching_results.tsv` (5,862,410 pairs) is a candidate. The official validator checks this and printed PASS [F: v10d].
- 184 S1 rows have no candidate (US 2, India 101, France 81). 34 France records appear in no candidate pair [M: `test_candidates.py`].

#### Pairs at each stage (test) [M: `test_candidates.py`]

Reduction ratio (RR) = 1 - (candidate pairs) / (all possible pairs). Two denominators are used:
- **Same-country:** S1 rows x records of the same country, summed over countries: 6,724,569,566,212 pairs.
- **All-country:** all S1 rows x all records: 17,272,751,604,416 pairs.

| Stage (test) | Pairs | Per S1 row | Per record | RR, same-country | RR, all-country |
|---|---:|---:|---:|---:|---:|
| Raw search output, production settings | 307,232,917 | 177.33 | 30.82 | 0.99995431 | 0.99998221 |
| Raw search output, wide US/India + production France | 469,773,900 | 271.15 | 47.12 | 0.99993014 | 0.99997280 |
| Stage-2 input | 10,687,724 | 6.169 | 1.072 | 0.99999841 | 0.99999938 |
| Pairs rescored by the cross-encoders (close calls + ranks 3-5) | 3,788,098 | 2.186 | 0.380 | 0.99999944 | 0.99999978 |
| **Final `candidate_pairs.tsv` (v10d)** | **12,249,116** | **7.070** | **1.229** | **0.99999818** | **0.99999929** |

| Final candidate file by country [M: `test_candidates.py`] | Pairs | Mean per S1 row | Median | 90th percentile | Max | RR, same-country |
|---|---:|---:|---:|---:|---:|---:|
| US | 4,345,438 | 6.55 | 6 | 9 | 72 | 0.99999828 |
| India | 5,837,449 | 7.21 | 7 | 11 | 525 | 0.99999847 |
| France | 2,066,229 | 7.96 | 7 | 12 | 1,329 | 0.99999445 |
| All | 12,249,116 | 7.07 | 7 | 10 | 1,329 | 0.99999818 |

- Long lists are rare: 299 S1 rows have more than 50 candidates (France 183, India 112, US 4), and 63 have more than 100 [M: `candidate_sources.py`].
  - The longest belongs to a generic France name ("Nantes Élémentaire SAS", 1,329 candidates). Many records point at such a name, and each record keeps only 1-2 candidates.
- The final candidate set is 3.99% of the production raw search output (12,249,116 / 307,232,917).
- There are 2.09 candidate pairs per matched pair.

#### Recall ceilings on the held-out half [M: `heldout_recall.py`]

Each candidate definition was rebuilt out-of-fold on the held-out half (1,102,482 S1 rows, 3,815,794 true pairs):
- every train record searched against all train S1 rows of its country;
- the wide search exists for test only, so its effect is not in these numbers.

| Candidate set (held-out) | Pairs per S1 row (US / India) | Recall, all | US | India | Records with address | Records without address |
|---|---|---:|---:|---:|---:|---:|
| Raw search output, production | 146.6 / 141.6 | 0.98509 | 0.98693 | 0.98235 | 0.99231 | 0.82881 |
| Stage-2 input | 4.888 / 5.029 | 0.97863 | 0.97965 | 0.97712 | 0.99160 | 0.69774 |
| + ranks 3-5 | 5.065 / 5.265 | 0.98042 | 0.98164 | 0.97859 | 0.99187 | 0.73241 |
| Second pipeline's stage-2 input alone | 4.862 / 5.009 | 0.98648 | 0.98638 | 0.98663 | 0.99809 | 0.73505 |
| **+ second pipeline (the final candidate definition)** | **5.346 / 5.675** | **0.98936** | **0.98944** | **0.98926** | **0.99853** | **0.79086** |

- The final definition keeps every true record of the S1 row for 96.3% of held-out S1 rows.
- It misses 40,586 of 3,815,794 true pairs. 35,213 of those misses (87%) are records without an address, mostly chain names repeated across many S1 rows.
- Where the true pairs are lost:
  - our production pipeline loses 1.49% in the search (0.98509) and 0.65% more at the stage-1 cut to 1-2 candidates per record (0.97863);
  - with the audited blocking the search loses only 0.37% (0.9963), so the cut becomes the larger loss (0.98648);
  - in both, the cut hits records without an address hardest (0.83 -> 0.70 for ours).
- Pairs per record: stage-2 input US 1.045 / India 1.074, final definition US 1.143 / India 1.212. The test file has 1.138 / 1.237, so the held-out sets match the test file in size per record.

#### Cost of the search (laptop CPU, Intel i5-12450H) [M: `compute_cost.py`, parsed from `work/*.log`]

| Operation | Volume | Rate |
|---|---|---|
| 4 searches, train (index build excluded) | 10,320,219 records, 319.0M pairs in 3,755 s | 2,748 records/s |
| 4 searches, test | 9,969,589 records, 307.2M pairs in 3,377 s | 2,952 records/s |
| wide search, test US/India | 8,534,596 records, 424.9M pairs in 7,431 s | 1,149 records/s |
| 55 pair features | 307.2M test pairs in 4,158 s | 73,890 pairs/s |
| stage 1 on every test pair (incl. reading from disk) | 307.2M pairs in 4,554 s | 67,469 pairs/s |

- Pair features take about 60 bytes per pair on disk (train 19.2 GB, test 18.6 GB, wide test 25.8 GB).

#### Template summary

- **Blocking keys used:**
  - TF-IDF over character 3-grams of the cleaned core name;
  - TF-IDF over word 1-2-grams of the cleaned address, and of core name + address;
  - fine-tuned multilingual-e5-small embeddings of "name | address" (non-Latin names; every record in the second pipeline);
  - the same-country constraint;
  - for records left unmatched in the countries without training labels (France), an exact address key (house number + street words) with a noise-only name test: the same-address rule, 223 pairs of the final candidate file.
  - These are followed by the stage-1 LightGBM filter that keeps each record's top 1-2 candidates (ranks 3-5 for close calls).
- **Candidate pairs generated:**
  - **12,249,116** = 7.07 per S1 row, 1.23 per record; RR 0.99999818 against same-country pairs.
  - Before the stage-1 filter the searches return 469.8M test pairs.
- **How you ensured true matches were not lost:**
  - each search was measured alone and in union at full density (not on a sample);
  - the caps were re-tuned after the density failure;
  - an embedding search covers non-Latin names;
  - no-address records get a longer name list;
  - close calls keep their stage-1 ranks 3-5;
  - a wide search covers busy US/India addresses;
  - the second pipeline's audited blocking adds the pairs our lists miss;
  - the held-out recall ceiling of the final candidate definition is 0.98936, and 0.99853 for records with an address.

---

## 4. Matching Model

**Features used:**
- **Name features** (stage 1):
  - TF-IDF cosine over character 3-grams, full vocabulary;
  - rapidfuzz ratio, token-set, token-sort and partial ratios;
  - Jaro-Winkler similarity;
  - ratio with legal forms kept;
  - exact core-name flag;
  - embedding cosine (non-Latin names);
  - name lengths;
  - number of S1 rows sharing the core name (chain size);
  - flags for web-domain names, "d/b/a" aliases and non-Latin script.
- **Address features** (stage 1):
  - character 3-gram cosine, word 1-2-gram cosine, name + address word cosine;
  - ratio, token-set and partial ratios;
  - house numbers: count on each side, common count, first number equal / contained / prefix, relative difference, digit edit distance;
  - overlap of alphanumeric unit tokens such as "8-9-1/14a";
  - address lengths;
  - missing-address and non-Latin-address flags.
- **Other:**
  - **Stage 1, competition among a record's candidates.** Margin to the best other candidate and rank, for name cosine, address cosine, name and address token-set, embedding cosine and name + address cosine. Also:
    - number of candidates;
    - number of candidates with name or address token-set >= 90;
    - which searches found the pair;
    - source (S2 or S3).
    - Total: 55 features.
  - **Stage 2, group features (76 = 55 + 21)** [M: feature list saved in `work/models/full_cons/result.json`]:
    - p1, its rank and margin;
    - S1-side competition: how many records point at this S1 row with p1 >= 0.5 (all, same source), their sum and maximum, this record's rank among them, the S1 row's candidate count;
    - sibling features: the best other record of the same S1 row, its p1, its name and address token-set against this record, same house number, same source, sibling missing;
    - consensus features: all records with p1 >= 0.5 at this S1 row, their mean and minimum name and address token-set, and the share and count with the same first house number.
  - **Stage 3:**
    - p1, p2;
    - the cross-encoder logit, its margin and rank among the record's close-call candidates;
    - p2 margin, number of close-call candidates, stage-1 rank.
  - **Cross-encoder input:** for each side, the lowercased text "name | address", cut to 160 characters; the pair is encoded as two segments of at most 128 tokens.
  - **US/India stacker (v10d), 24 features:**
    - per-family probabilities (e5-small, bge-reranker-v2-m3, e5-large), our family blend, the second pipeline's probability, the v10a blend;
    - both pipelines' p1 and p2, which pipeline scored the pair and whether it was a transformer close call there;
    - rank and margin of the blend within the record;
    - country: its position in the list of countries with training labels, ordered by train S1 rows (US 0, India 1);
    - record and S1 row without address;
    - signed and absolute house-number difference;
    - exact core-name equality;
    - differences our blend minus second pipeline, and e5-small minus bge.

**Model type:**

| Stage | Model | Trained on | Output |
|---|---|---|---|
| Stage 1 | LightGBM, 255 leaves, learning rate 0.08, 600 rounds; 3 fold models (out-of-fold scores) + one full-sample model for test | 11.5M sampled pairs: all 3.76M positives, 15% of hard negatives (rank <= 2 by name cosine or address token-set, or name token-set >= 80), 1.5% of other negatives [E: Run 9, run 7] | p1 for every searched pair |
| Stage 2 | XGBoost (GPU), depth 8, learning rate 0.05, 500 rounds; the 3 fold models are averaged on test | each record's stage-2 input, 50% of records per training fold | p2 |
| Cross-encoders | 3 fold models per family, 1 epoch, binary cross-entropy, AdamW with a one-cycle schedule | fold model k trains on all labelled close calls (with ranks 3-5 and records spanning both halves) whose record is in another fold: 1.68M pairs | logit per close-call pair |
| Stage 3 | XGBoost, depth 6, learning rate 0.05, one per family | out-of-fold cross-encoder logits of the labelled close calls | p3; on test each fold model's scores go through stage 3 and are averaged |
| Family blend (US/India) | 0.3 x e5-small family + 0.7 x bge family, the best of the weights 0.3/0.7, 0.5/0.5, 0.7/0.3 on held-out; bge alone scored the same (0.98951 vs 0.98950) | | p for our pipeline |

Cross-encoder families [R: code README steps 8-12; E: e5-large as a third cross-encoder family]:

| Family | Parameters | Training | Where it ran | Used for |
|---|---|---|---|---|
| multilingual-e5-small (MIT) | 117.7M | word-embedding table frozen, gradient checkpointing, batch 32, lr 3e-5 | laptop RTX 3050 4 GB | US/India (3 folds); France (two half models A and B) |
| multilingual-e5-base (MIT) | 278.0M | 16-bit frozen word table, batch 16, 420k rows (model A2) | laptop | France rows only |
| bge-reranker-v2-m3 (Apache-2.0) | 567.8M | full fine-tuning, batch 64, lr 2e-5, 26,000 steps per fold | RTX A6000 48 GB (Bhavya Jain) | US/India (3 folds) |
| bge-reranker-v2-m3, second pipeline | 567.8M | full fine-tuning, 1M close-call pairs per fold | RTX PRO 6000 Blackwell 96 GB (lab server) | the second pipeline's 3 folds |
| multilingual-e5-large (MIT) | 559.9M (computed from its published configuration, B.7) | full fine-tuning, batch 64, lr 2e-5 | RTX A6000 (folds 0-1), lab server (fold 2) | input `c` of the v10d stacker |

Held-out, all close calls rescored [E: 3-fold transformers; e5-large; M: best-rule scores in `work/ce_b2/rule_<family>folds.json`]:
- e5-small family 0.98903; bge family 0.98951; blend 0.3 / 0.7 0.98950 (v7p);
- e5-large family alone 0.98924; the three families blended equally 0.98946.
- The fixed blends therefore do not use e5-large. The v10d stacker takes it as one of 24 inputs.

#### Second pipeline

[E: v8 and v9 sections; R: code README step 17]
- Same code: `pipeline.py`, `rerank.py`, `ce_folds.py`, with `BER_V8=1` (audited blocking and cleaning, section 3).
- Stage 1: XGBoost (the default backend of `pipeline.py`: depth 10, 1,000 rounds). Stage 2: XGBoost with the same group features.
- Cross-encoders: 3 bge-reranker-v2-m3 fold models on 1M close-call pairs each, then stage 3.
- Held-out: GBDT 0.98472; with the transformers 0.99112 (halves) / 0.99183 (all rescored).
- Test close calls: 2.65M pairs in 2.17M records (21.8%).
- It ran on an NVIDIA RTX PRO 6000 Blackwell 96 GB shared server; the transformer stage took 2.45 h.

#### Combining the two pipelines (US/India)

1. **v9y: missed pairs.**
   - The second pipeline's matched pairs are added when the pair is absent from our candidate lists and our file leaves the record unmatched: 16,421 pairs (India 12,939, US 3,482) [L: v9y].
2. **v9zm: list-mover fix** [R: `movers/build_movers.py`; F: v9zm].
   - The wide search changed the candidate lists. Stage 2's list-dependent features (counts, ranks, margins) then moved some pair probabilities although the texts had not changed.
   - Where the wide build and the old build disagree on a pair, the decision goes to a transformer-only probability `pt`. `pt` is the mean over the e5-small and bge families of P(true | fold-mean logit), calibrated per country on train out-of-fold close calls; it depends only on the two texts.
   - Remove a pair that only the wide build accepted if `pt < 0.5` (tier 1, the wide build did not rescore it) or `pt < 0.2` (tier 2).
   - Restore a pair that only the old build accepted, for a record that is otherwise unmatched, if `pt >= 0.8` / `0.95`.
   - Result: -839 / +832 pairs; estimated +0.00012 to +0.00015 public LB from a label check on a rebuilt train sample (estimate) [L: v9zm].
3. **v10a: per-pair blend** [R: `blend_second.py`; F: v10a].
   - p = 0.6 x ours + 0.4 x second pipeline where both scored the pair; otherwise the one that scored it. Then the list-mover overrides, then the decision rule.
   - The weight 0.4 was chosen on held-out: ours alone 0.98950, second pipeline alone 0.99183, blend 0.99205.
   - Against a held-out simulation of v9y the blend gained +0.00019 on US/India (each fold +0.00017 to +0.00021), about +0.00016 on the leaderboard since US/India is about 85% of it; the public LB moved by +0.000198.
4. **v10d: stacker** [R: `stack/README.md`; F: v10d; E: Second final-upload hunt].
   - LightGBM, 15 leaves, 300 rounds, learning rate 0.05, on the 24 per-pair features above, over every pair either pipeline scored.
   - Features that depend on the length of the candidate lists were dropped (the record's candidate count, and the rank and margin within the S1 row), because test candidate lists are wider than held-out ones [R: `stack/README.md`; the `DROP` list of the `fit.py` commands].
   - Cross-fitting:
     - the held-out half is split in two by S1 id; models fitted on one part score the other;
     - two more models are fitted on the training half;
     - the test probability is the mean of the 4 models.
   - The list-mover overrides are applied after the model, then the decision rule.
   - Held-out: 0.99205 -> **0.99226** (+0.000209, about 13 standard errors). Both parts of the held-out half gain (+0.000216 / +0.000202), as do both countries (US +0.000226, India +0.000184).
   - Its held-out additions were 84.6% true; 43% of its removals were false pairs.
   - On test: US +3,733 / -841 pairs, India +3,288 / -434.
   - What it relies on, split gain averaged over the 4 models [M: `stack_importance.py` on `lgb_models_avg.pkl`]: margin of the blend within the record 47.6%, the v10a blend 43.7%, the second pipeline's probability 8.2%, all other features 0.6% together (e5-large 0.08%).

**Threshold selection method:** expected-F0.5 set selection (`pipeline.decide_expf`), chosen on held-out macro F0.5.
- For a set of k predicted records, F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP) = 1.25 TP / (k + 0.25 T), where T is the number of true records.
- Each record goes to its highest-probability S1 row. For each S1 row, the records with p >= floor are sorted by p.
  - Expected TP of the top k is the sum of their p. T is estimated by the sum of all their p.
  - The rule picks the k that maximises 1.25 x (sum of top-k p) / (k + 0.25 x sum of all p).
  - The S1 row gets an empty list when the probability that none of them matches (product of 1 - p) is higher than that best value.
- The rule was chosen among 17 candidates by held-out macro F0.5 [E: Transformer versions]:
  - plain thresholds 0.4 to 0.9;
  - floors 0.3 / 0.5 with exponents 1 / 1.5 / 2 applied to p.
- The final setting is floor 0.5, exponent 1, for US/India and for France (France keeps the rule chosen with its transformer set).
- Rules tried and not kept:
  - per-segment thresholds: +0.0000 [E: v2 pipeline on S10];
  - isotonic recalibration with per-country rules: +0.0002 held-out, left out because it added about 56k borderline test matches in the band where test has twice train's look-alike share [E: v6 on FULL data].

#### France: decisions without labels

In the code, the rules of this section apply to the countries without training labels: the test Source 1 countries that the train Source 1 file does not have (`common.unlabelled_countries()`, read from the data). In this test set that is France alone. No country name is written in the pipeline code (Appendix B.8).

France rows use the same stage 1 and stage 2. They keep the halves transformer set of v7ens:
- 0.7 x the e5-small models A + B, 0.3 x the e5-base model A2;
- the France score of this set was measured on the leaderboard: about 0.952 with the min rule and the legal-form veto, before the later rules;
- a new stage 3 changes France rows even under the same rule: v7g_num changed 12,477 France S1 rows that way [E: 3-fold transformers].

A France change is measured on the leaderboard by an upload that changes only France rows. Then

    France F0.5 change = public LB change / 0.1497

where 0.1497 is France's share of test S1 rows. This assumes the public subset has the same country mix.

**Rules used** (in build order):

| Rule | Pairs | Label-free evidence | Leaderboard effect |
|---|---|---|---|
| Legal-form veto: p = 0 when both raw names carry a legal form (SARL, SAS, SA, EURL ...) and none is shared | 17,934 France pairs with p2 >= 0.3 at v7a [E: Held-out loss breakdown] | such conflicts are true in 5.2% of US and 0.0% of India labelled pairs [E: France checks] | not isolated (v7a was not uploaded) |
| Min rule: France p = min(p2, p3); the transformers may only lower a France probability | all France close calls | records the transformer adds change house number 34% of the time, records both keep 1.6% [E: France label-free audit]. This basis was weak: it compared with easy pairs, not with true pairs [E: France checks]. The leaderboard settled it. | removing it (v7i) cost France -0.0035 (LB 0.982636 vs 0.983159) |
| Descriptor-word veto: p = 0 for accepted pairs whose record adds or swaps in a descriptor word (amicale, comite, ecole, club, centre ...), synonym and stem swaps excluded | 22,436 (21,636 with the same house number; 230 are Club <-> Ecole swaps [M: `set_summary.py`, `desc_veto_examples.py`]) | all-lowercase share of the set 3.75% [L: v9a]: at or above the decoy rate (2.4-3.0%) and far above true records (0.03-0.22%), so about 0% true | inside the v9y upload (+0.0064 with other changes); predicted France +0.017 to +0.020 [L: v9a] |
| Noise-word additions: p = max(p, 0.95) for rejected same-street, same-number pairs whose added or swapped word is a noise word (et fils, & associes, cie, services ...) or groupe / developpement / france | 9,800 | lowercase test about 0.96 true; a fit to the France probe results gave 0.77-0.81 [L: v9b] | inside v9y |
| False-accept veto: p = 0 | 1,145 in three tiers [M: `set_summary.py`]. A (846): legal form added and house number up by 1-20, the decoy signature. B (232): word added, swapped or mistyped and number up by 1-20. C (67): all-lowercase record with a name change. | label-free estimate of the true share, saved with the set (column `t` of `sets/fp_veto_set.parquet`) [M: `set_summary.py`]: A 0.0; B 0.60; C 0.42. Tier A is all-lowercase 3.4%, the decoy rate [M: `set_summary.py`]. [L: v9e] | inside v9y; predicted France +0.0007 [L: v9e] |
| Recoveries: p2 = 0.95 | 5,883 [M: `set_summary.py`]: earlier missed-match tiers A1-A4 (4,150; estimated true 0.91-0.96); acronym, dotted-legal-form or web-domain copies at the same address (1,731; 0.85-0.95); exact-name twins (2). Mean estimated true share 0.93. | per-group estimated true share saved with the set (column `est` of `sets/recall_add_set.parquet`) [M: `set_summary.py`; F: v9z]. For the 4,150 earlier missed-match pairs: lowercase test about 0.9 true, a leaderboard-realism fit 0.75-0.8 [L: v9f] | v9zm +0.000405 together with the list-mover fix. The France part is about +0.00026 to +0.00029 LB, France +0.0017 to +0.0019 (estimate). |
| Same-stem descriptor veto (sport -> sportive) | 73 | as the descriptor veto | inside v9zm |
| Second-pipeline France matches outside our lists, for records we leave unmatched. Only pairs that differ by generator noise are kept: no word-level name change, no legal-form change or addition, house number not moved up, second-pipeline p >= 0.8. | 4,825 | US/India analog on labels: 98.4% true (28,113 pairs), 98.8% under the filter; 0 kept non-domain names are all-lowercase [R: `france_recall.py`] | v10b: +0.000309 LB = France +0.00206 (predicted +0.0018) [L] |
| Three verified France sets [F: v10c] | 802 | typos that garble the S1 word with its letters (416; US/India analog 99.3% true); the same-address rule, one S1 row at the address and the name differs only by acronym or noise (298; train precision 99.85% on 651 pairs; rule in `same_address.py`, section 3); "&" written as "et" or "+" (88) | inside v10d; expected +0.00003 to +0.00005 LB |
| Street veto: p = 0 when name and house number are equal but the street is completely different, and both bge families score below 0.3 | 256 | US/India analog: 1 of 87 true [F: v10d] | inside v10d |

**Rules the leaderboard or the labels refuted**

| Hypothesis | Test | Result |
|---|---|---|
| Letting the transformers add France matches (v7i) | France-only upload | France -0.0035 [L] |
| France rejects true records whose house number changed; add them up to half the US true rate (v7j, +21,155 records) | France-only upload | France -0.0163 [L]. In France a changed house number marks a decoy. |
| Restore same-address pairs the transformer lowered (v7m, +21,581 France pairs; 16.7k of the 19.8k restored same-address pairs were descriptor swaps) | France-only upload | France -0.0091 [L]. Those pairs were about 40% true, which led to the descriptor veto. |
| French address cleaning (dotted legal forms, street abbreviations, "et" for "&") (v6) | upload | LB fell (0.967332 vs 0.9700). A test-like check put the consensus features +0.0015, so the drop most likely came from the cleaning [E: Test-like split tl2]. |
| Descriptor-swap veto as first written, 64,756 pairs (v7h) | US/India labels | such swaps are 98.9% (US) / 98.2% (India) true. Not uploaded. [E: France checks] |
| Per-pattern recalibration of France to US true rates | count bound | would accept 1.01M France records against about 0.88M expected (3.4 per S1 row) [E: France checks] |
| France house-number veto (v7k) | label-free audit | the first-number parser picks apartment and floor numbers ("Appartement 22"); uploaded once inside v7g_num together with another France change (not attributable); not in the final file [E: Transformer at test-like density; L: v7g_num] |
| France pairs both pipelines scored but only the second accepted | US/India analog | 69.6% true, and France descriptor classes about 0-10% true; not used [E: Final-upload hunt] |
| France vetoes where both bge families score low (13,092 pairs) | v9zm leaderboard result (v9zm carried the v9z France rows; v9z itself was not uploaded) | 83% are "X <descriptor> SARL" -> "... Developpement / Groupe / & Associes". The US-trained models treat these as look-alikes, but the v9zm result shows them about 90% true. Only the 256-pair street veto was kept [E: Second final-upload hunt]. |

The lesson: a rule's truth rate can differ between countries (a descriptor swap is noise in US/India but marks a decoy in France). What transferred were the instruments: the lowercase share and the house-number direction are properties of how the generator makes decoys, and they hold in both labelled countries.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - held-out **0.99226** on US/India (held-out half at full density, all close calls rescored, final v10d stacker);
  - public leaderboard of the final file **0.990565**;
  - France has no labels, so no held-out number exists for it. Its leaderboard-based estimate is about 0.98 from v9y on (below).

#### Ablation: every version on the path to the final file

| Version | Change against the previous row | Held-out macro F0.5, US/India | Public LB | Change on public LB |
|---|---|---|---|---|
| rule | best (name token-set + address token-set) / 200 >= 0.8 | 0.8925 (10% slice) | - | - |
| v1 | TF-IDF shortlist + LightGBM, 40 features | 0.9753 (10% slice; not comparable) | not uploaded | - |
| v2 | word-level blocking valid at full density; two-stage LightGBM; expected-F0.5 rule | 0.97904 | 0.9677 | first upload |
| v3 | + sibling features, XGBoost stage 2 | 0.98015 | 0.9700 | +0.0023 |
| v4 | trained on resampled "test-like" data | 0.97753 (other validation) | 0.961357 | -0.0086 vs v3; dropped |
| v6 | + consensus features; + French cleaning | 0.98151 | 0.967332 | -0.0027 vs v3; cleaning dropped |
| v7 | + e5-small cross-encoder on close calls, stage 3 | 0.98794 (halves) | - | - |
| v7b | two cross-fitted e5-small models | 0.98805 (halves) | - | - |
| v7d | + stage-1 ranks 3-5 rescored | 0.98811 (halves) | - | - |
| v7ens | + e5-base model, blend 0.7 / 0.3; France: min rule + legal-form veto | 0.98813 (halves) | 0.983159 | +0.0132 vs v3 |
| v7g | e5-small and e5-base, 3 fold models each | 0.98845 (halves) / 0.98911 (all) | 0.983103 (as v7g_num, with an untested France change) | -0.000056 |
| v7p | e5-small and bge, 3 folds each, blend 0.3 / 0.7; wide US/India test search | 0.98878 (halves) / 0.98950 (all) | - (inside v9y) | - |
| v9y | + 16,421 second-pipeline pairs; France: descriptor veto, noise-word additions, false-accept veto | 0.98950 for its v7p base; the wide search and the added pairs are not measurable on held-out | 0.989563 | +0.0064 vs v7ens |
| v9zm | + 5,883 France recoveries, -73 France pairs; US/India list-mover fix | same | 0.989968 | +0.000405 |
| v10a | US/India p = 0.6 ours + 0.4 second pipeline | 0.99205 (all) | 0.990166 | +0.000198 |
| v10b | + 4,825 France pairs from the second pipeline | 0.99205 (France only changed) | 0.990475 | +0.000309 |
| v10c | + 802 France pairs from three verified sets | 0.99205 | not uploaded (included in v10d) | - |
| **v10d** | US/India stacker over both pipelines; -256 France pairs | **0.99226 (all)** | **0.990565** | +0.00009 vs v10b |

Sources: [E] sections Runs, Run 9, Submissions plan 25 Sep, v6 on FULL data, Transformer versions, 3-fold transformers, v9 transformer stage, Second final-upload hunt; [L] for every public score.

- Second pipeline alone: GBDT 0.98472; with its bge folds 0.99112 (halves) / 0.99183 (all).
- On the leaderboard its US/India rows scored below ours: v9x = its US/India + our v9e France = 0.989266, against v9y 0.989563. That is about 0.00035 lower on US/India F0.5 [L: v9x].
- After v10a the team was at public rank 28 (top score 0.991829) [L: v10a].

#### Probe uploads: measuring France with the leaderboard

An upload with every France row emptied scores France S1 rows as 1 only where they truly have no match. So the difference to a full upload gives France's score [L].
- The formulas assume France's singleton share equals train's 5.6%:
  - France = (LB_full - LB_probe) / 0.1497 + 0.056;
  - US/India = (LB_probe - 0.15 x 0.056) / 0.85.

| Upload | Public LB | What it measured |
|---|---|---|
| v5 = v3 with France emptied | 0.838 | France about 0.937, US/India about 0.976 (v3) |
| v7ens_frempty = v7ens with France emptied | 0.848792 | France about 0.952-0.954 (0.952 in `submissions/LOG.md` with weights 0.15 / 0.85; the formula above gives 0.954), US/India about 0.9887 (v7ens); France was about 70% of the gap to the top score then (0.990621) |
| v7i (France only: transformers may add) | 0.982636 | France -0.0035 |
| v7j (France only: number-changed additions) | 0.98071 | France -0.0163 |
| v7m (France only: restore same-address pairs) | 0.9818 | France -0.0091 |
| v9x (US/India of the second pipeline) | 0.989266 | its US/India about 0.00035 below ours |
| v10b (France only: 4,825 additions) | 0.990475 | France +0.00206 |

- **France from v7ens to v9y** (estimate). The public score rose by 0.0064. The measured or estimated US/India gains cover about 0.85 x 0.002 = +0.0017 of it:
  - held-out +0.00065 on the halves protocol for the 3-fold e5-small and bge families;
  - an estimated +0.0011 to +0.0015 for the wide search.
- The second pipeline's 16,421 pairs were not measured alone. If they added nothing, France rose by about (0.0064 - 0.0017) / 0.15 = +0.031, from about 0.952 to about 0.98.

#### Does held-out predict the leaderboard?

| Change | Predicted | Observed on public LB |
|---|---|---|
| v3 sibling features | held-out +0.0011 | +0.0023 [E: Submissions plan 25 Sep] |
| v3 -> v7ens (consensus features + cross-encoders), US/India | held-out +0.0080 (0.98015 -> 0.98813) | US/India +0.0127 (probes) |
| v10a blend with the second pipeline | held-out +0.00019 on US/India (about +0.00016 LB) | +0.000198 |
| v10b France additions | LB +0.00027 (from the US/India analog) | +0.000309 |
| v10d stacker and France changes | LB about +0.0002 | +0.00009, about 45% of the expected gain [E: Second final-upload hunt] |
| second pipeline vs ours, US/India | held-out +0.0023 in its favour (0.99183 vs 0.98950) | 0.00035 against it (v9x vs v9y) |

- Within one pipeline, held-out moved the leaderboard in the predicted direction. Changes that handle decoys gained more on test (v3, v7ens), where decoys are more frequent.
- Across pipelines the absolute held-out scores were not comparable, so blends were chosen by held-out gains on the same rows.

#### Held-out loss breakdown

Each number is the macro F0.5 recovered if that mistake type alone were fixed.

After the first cross-encoders (v7b, 0.98805) [E: Remaining held-out loss after the transformer]:

| Mistake | Points | Pairs |
|---|---:|---:|
| true record never in the stage-1 top 2 | 0.00681 | 81,531 (records without address: 50,893 pairs, 0.00416) |
| right S1 row found, rejected by the decision rule | 0.00258 | 31,933 |
| another S1 row chosen | 0.00113 | 13,842 |
| record with no true S1 row merged | 0.00083 | 2,999 |
| record of another S1 row merged | 0.00060 | 2,285 |

After the v10a blend (0.99205) [E: France recall from the second pipeline]:
- The total loss is 0.0080. **Records without an address cost 0.0060 of it:**
  - not in any candidate list 0.00285;
  - another S1 row chosen 0.00158;
  - rejected by the rule 0.00153.
- Records with an address cost the remaining 0.0020.

**Common false positives (wrong merges):**
- **Decoys with a shifted house number (US/India).** Same name and street, house number moved up by a few units: "9692 Diamond Rd" vs S1 "9687 Diamond Rd", "657 vs 652 39th Ave" [E: Test-side EDA].
  - Some decoys look fully true: "Consolidated Médical Studios \| 31 Laurel Circle" vs S1 "... \| 315 Laurel Circle", accepted with p = 0.99 [E: v2 pipeline on S10].
  - After the transformers, merged records with no true S1 row cost 0.00083 and merged records of another S1 row 0.00060 (table above).
- **Same address, unrelated name.** A web-domain name at the right address was accepted with p = 0.94 [E: Error analysis of run 2].
- **France descriptor swaps: the "Club" vs "Ecole" look-alike** [M: `france_examples.py`, `desc_veto_examples.py`].
  - Record S3-551485150 "Nantes Ecole  SARL \| 26 Rue Louis Blanc, Nantes, Loire-Atlantique" vs S1-822548516 "Nantes Club SARL \| 26 Rue Louis Blanc, Nantes, Pays de la Loire".
  - Same street and number, same legal form, only the descriptor word differs. The models gave p2 = 0.9995, and no S1 row named "Nantes Ecole SARL" exists at that address.
  - In US train labels such swaps are 98.9% true. In France they are decoys: the lowercase test says about 0% true, and restoring them cost France -0.0091 (v7m).
  - The descriptor veto removes 22,436 such pairs, 230 of them Club <-> Ecole.
- **France legal-form twins.** SARL vs SAS on otherwise equal records; handled by the legal-form veto.
- **France same name and number, different street** [M: `france_examples.py`]: record "Bordeaux Primaire SARL \| 32 R. SAINTE-LCUE, BORDEAUX" vs S1 "Bordeaux Primaire \| 32 Rue Mondenard, Bordeaux"; removed by the v10d street veto (256 pairs).

**Common false negatives (missed matches):**
- **Records without an address in a chain.** The record gives only a name that several S1 rows share ("Perfect Food Pvt Ltd" at several addresses) [E: v2 pipeline on S10].
  - For no-address records whose true S1 row's core name is shared by 2-60 S1 rows, a random pick is right 26.2% of the time; the closest raw name (legal form and punctuation kept) is right 42.3% [E: Shortcut checks].
  - That is too weak to pay under a precision-weighted metric, so these records stay unmatched. Records without an address cost 0.0060 of the 0.0080 held-out loss (above).
- **Right candidate rejected because the house number changed or is missing** ("11" vs "127 Lindsey Avenue") [E: Run 9].
  - Before the transformers: changed number 0.00290, no number 0.00196 of this mistake type's 0.00692 [E: Held-out loss breakdown].
- **True S1 row not among the stage-1 top 2** (0.00681 at v7b). The ranks 3-5 of close calls and the second pipeline's lists were added for this.
- **France noise-suffix names.** "X <descriptor> SARL" -> "... Developpement / Groupe / & Associes" is about 90% true in France (v9zm upload, which carried the v9z France rows), but the US-trained transformers score it low [E: Second final-upload hunt]. The noise-word additions (9,800 pairs) recover part of it.
- **France acronyms and web domains.** "SDS" = "Securite Darts Sport"; "PC" = "Paranormal Club"; "nantesclubsas.com" [E: Cleaning bugs found; R: `france_recall.py`].
  - About 11k France records are acronyms of their S1 name (53.6% of short-name records have an S1 row with equal initials at the same number, against 16.1% by chance).
  - The recoveries and the v10c same-address rule add the ones the checks accept.

---

## 6. Conclusion

- A bounded top-k search with a learned filter keeps 7.07 candidates per S1 row (reduction ratio 0.99999818), with a held-out recall ceiling of 0.98936.
- A GBDT cascade with group features, fold-trained cross-encoders on the uncertain 23.5% of records, and a stacker over two pipelines reaches 0.99226 held-out macro F0.5 on US/India and **0.990565** on the public leaderboard.
- For France, which has no labels, each rule was checked with a label-free test built on a data-generator artefact, with the analogous pairs in labelled US/India data, or both. Two rules in the final file were confirmed by uploads that changed only France rows (the min rule by v7i, the v10b additions); the others reached the leaderboard inside bundled uploads (v9y, v9zm, v10d).

Lessons, with the evidence for each:
- **Validate at full density** (every S1 row of the country in the index, as on test). A 10% slice hid a recall collapse to 0.7968. After re-tuning at full density, recall was 0.9851, and the audited settings reached 0.9963.
- **Resampled training data must be compared with test on every feature that depends on the candidate list.** Deleting pairs to imitate test halved the candidate lists (15.9 vs 31.3 per US record) and cost -0.0086 on the leaderboard (v4).
- **The largest gains came from transformers on close calls** (+0.0070), **from comparing records with each other** (group features, +0.0025) **and from a second pipeline with audited blocking** (+0.0032 at the GBDT level; blend +0.00255 over ours).
  - Scoring every pair with the transformer adds little over close calls only: 0.99362 vs 0.99351 in a controlled test (Appendix B.4).
- **The remaining loss is mostly records without an address** (0.0060 of 0.0080 held-out); 0.00285 of it is true S1 rows that are in no candidate list. A second pipeline with a different blocking raised the recall ceiling more than any measured change to our own lists (+0.0089, against +0.0018 for ranks 3-5).
- **For an unlabelled country, check each rule on the labelled countries first, but do not assume the rule transfers.**
  - What transferred were measurement instruments: the lowercase share and the house-number direction.
  - A France change uploaded alone with US/India rows fixed (v7i, v7j, v7m, v10b) could be measured; changes uploaded together (v7g_num, and the France parts of v9y, v9zm and v10d) could not be attributed.
- **Held-out predicts the leaderboard within a pipeline, not across pipelines.** The v10a blend moved the public score by +0.000198, slightly more than its held-out gain implies (0.85 x 0.00019 = +0.00016). The last upload (v10d: the stacker plus the v10c/v10d France changes) transferred at about 45% of the gain expected for both together.

---

## Appendix

### A. Code Artefacts

The submission zip holds `code/business_entity_resolution/` with `README.md` (every command, in order, with run times and machines), `requirements.txt` (the laptop's pinned versions: Python 3.12.10, polars 1.44.2, scikit-learn 1.9.1, sparse_dot_topn 1.2.0, rapidfuzz 3.14.6, LightGBM 4.7.0, XGBoost 2.0.3, torch 2.7.1+cu118, transformers 5.17.0), `sets/` and `src/`.

**Structure**

| Path | Contents |
|---|---|
| `src/common.py`, `normalize.py`, `learn_maps.py`, `prep.py` | folders (`BER_DATA`, `BER_WORK`, `BER_OUT`), TSV reading, the country sets read from the data, macro F0.5; cleaning with maps learned from train matches |
| `src/embed.py` | fine-tuned multilingual-e5-small; vectors for the embedding search |
| `src/candidates.py` | the four searches and the 55 pair features |
| `src/pipeline.py` | `build` / `train` / `predict`: stages 1 and 2, grouped 3-fold validation, expected-F0.5 decision |
| `src/topk5.py`, `src/rerank.py`, `src/ce_folds.py`, `src/bundle.py` | ranks 3-5; close calls, cross-encoder training and scoring, stage 3; the 3-fold runner for a GPU machine and its input bundle |
| `src/blocking_b2.py` | wide US/India test search |
| `src/blend.py`, `src/fr_minrule.py`, `src/finalize.py` | family blend and rule choice; min rule and legal-form veto for the countries without training labels (France); per-country decision |
| `src/assemble_final.py`, `src/make_candidates.py`, `src/check_submission.py` | final matching file, candidate file, rule check |
| `src/movers/build_movers.py`, `src/blend_second.py`, `src/france_recall.py`, `src/apply_pair_sets.py` | v9zm list-mover fix, v10a blend, v10b France recall, v10c/v10d pair sets |
| `src/same_address.py` | the same-address rule (section 3): regenerates the 298 pairs of `sets/fr_same_address_safe.parquet` from the cleaned test files and v10b |
| `src/stack/` | v10d stacker: `build_ho.py`, `fit.py`, `score_avg.py`, `merge_models.py`, `build_test.py`, `apply_test.py`, `feats.py` (commands in `stack/README.md`) |
| `src/build_final.py` | rebuilds the final files from the saved score files and `sets/` (below) |
| `src/france_fix/` | the France analyses that chose each France set (generator change census `artifacts/census/`, look-alike and missed-match hunts, leaderboard fits, `v9z/`, `final_hunt/`); run once, interactively; index in `src/france_fix/README.md` |
| `src/runners/` | shell drivers of the lab-server run (`run_v9.sh` = the second pipeline) and of the unattended laptop runs |
| `src/experiments/` | experiments not needed for the final files (blocking audit, error analyses, deep-learning matcher test, test-like training); `measure/` holds the scripts behind the [M] numbers of this document |
| `sets/` | the 13 fixed pair sets and the stacker model file (`lgb_models_avg.pkl`); row counts, md5 and the script that made each one in `sets/README.md` |

**Entry points, from data to both outputs** (full commands in `README.md`)
1. **Cleaning and embedding:** `learn_maps.py` -> `prep.py` -> `embed.py train|encode`.
2. **Blocking and features:** `pipeline.py build train full` -> `BER_BACKEND=lgb pipeline.py train full` (stage 1) -> `BER_BACKEND=lgb BER_BACKEND2=xgb pipeline.py train full cons` (stage 2) -> `pipeline.py build test test` -> `pipeline.py predict full_cons test`.
3. **Close calls and cross-encoders:**
   - `rerank.py select` -> `topk5.py` -> `rerank.py extras`;
   - France transformer set: `rerank.py train|score|stage3`, `blend.py`, `fr_minrule.py`;
   - `ce_folds.py`: 3-fold e5-small on the laptop; 3-fold bge and e5-large on a 48 GB GPU.
4. **Wide search and US/India scores:** `blocking_b2.py build|predict|rows` -> `rerank.py score|stage3` -> `blend.py`.
5. **Second pipeline:** `src/runners/run_v9.sh`, the same scripts with `BER_V8=1` on the lab server.
6. **Final files:** `python build_final.py` rebuilds v9z France, v10a, v10b, v10c and v10d from the saved score files of steps 1-5 and `sets/`, then runs `check_submission.py` and the official validator.
   - `output/matching_results.tsv` is written by `assemble_final.py`; `output/candidate_pairs.tsv` by `make_candidates.py`.

**Reproduction check** [M: `src/logs/build_final.log`, 27 Sep 2026]
- `build_final.py` ran from an empty build folder in 304 s on the laptop. Each step runs in its own process; the largest (`stack/build_test.py`) used 5.46 GB [M: `src/logs/build_final_memory.txt`, sampled by `peak_memory.py`].
- The md5 of the rebuilt v10a, v10b and v10c files and of both final files equal those of the uploaded files: `matching_results.tsv` `18412329111b5d9c43df3a58df0574d1`, `candidate_pairs.tsv` `a2ddcb5d26ede94f780fd2c4d87519a7`.
- `check_submission.py` and the official validator (`--check-ids`) printed PASS.
- 28 Sep 2026: after the same-address check was added and the country names were removed from the code, `build_final.py` ran again from an empty build folder [M: `src/logs/build_final_28sep.log`]. `same_address.py` regenerated the 298 shipped pairs in 16 s, all five md5 values were unchanged, and both checkers printed PASS.

**What is exact and what depends on saved outputs**
- **Exact:** every step inside `build_final.py`, given the saved score files. It calls the same scripts that built each version.
- **Close, not identical, on a full rerun:**
  - GPU transformer training is not bit-for-bit deterministic. Retrained cross-encoders give close but not identical scores.
  - The second pipeline's output was produced on the lab server and received as files.
- **Shipped as saved outputs:**
  - the France pair sets, chosen by the interactive analyses in `src/france_fix/`. The same-address set is also regenerated by `same_address.py` inside `build_final.py`, which stops if the regenerated pairs differ from the shipped ones (on 28 Sep 2026: 298 pairs, identical);
  - the mover tiers, which `movers/build_movers.py` rebuilds identically from the saved v7p, v7q and v9y files;
  - the stacker models: the `fit.py` commands in `stack/README.md` were reconstructed from the saved models' feature lists, not logged at the time.

### B. Additional Results

#### B.1 Cross-encoders on the same held-out close calls (AUC)

Fold k = labelled close calls whose record is in fold k, records whose candidates span both S1 halves left out. No compared model trained on these pairs.

| Fold (pairs) | e5-small halves models A+B (v7ens) | e5-small, 3 folds | e5-base, 3 folds (800k training rows) | mean of e5-small + e5-base logits | bge-reranker-v2-m3, 3 folds | e5-large, 3 folds |
|---|---:|---:|---:|---:|---:|---:|
| 0 (557,355) | 0.98875 | 0.99121 | 0.99007 | 0.99168 | 0.99392 | 0.99327 |
| 1 (551,093) | 0.98891 | 0.99144 | 0.99081 | 0.99206 | 0.99412 | 0.99358 |
| 2 (551,325) | 0.98830 | 0.99084 | 0.98994 | 0.99141 | 0.99373 | 0.99313 |

- Sources: the first four columns are from [E: 3-fold transformers]. The bge and e5-large columns are [M: `ce_auc.py`], computed from the saved out-of-fold score files (`work/ce_b2/train_ce_<family>f<k>.parquet`); the same computation reproduces the e5-small column exactly.
- On every out-of-fold row of each fold (837k-846k pairs, including ranks 3-5 and records spanning both halves): e5-small 0.98723 / 0.98731 / 0.98676, bge 0.99022 / 0.99034 / 0.98995, e5-large 0.98926 / 0.98952 / 0.98922 [E: e5-large].
- Same model size, twice the data: the 3-fold e5-small model beat the two half models on the same 280,159 pairs, AUC 0.98865 -> 0.99113, log loss 0.13020 -> 0.11716 [E: early check 16:30].
- The e5-base model trained on 800k rows ranks below e5-small trained on all 1.68M rows on every fold. Capping the training rows cost more than the larger model gained.

#### B.2 Stage 3 and blends, held-out macro F0.5

| Setting | Halves | All rescored |
|---|---:|---:|
| GBDT only (stage 2 + rule, no transformer), v6 model | 0.98151 | 0.98151 |
| e5-small halves A+B (v7b) | 0.98805 | - |
| v7ens: 0.7 (A+B) + 0.3 e5-base A2 | 0.98813 | - |
| e5-small 3 folds (v7f) | 0.98838 | 0.98903 |
| e5-base 3 folds | 0.98826 | 0.98890 |
| e5-small + e5-base, stage 3 per family, blend 0.7 / 0.3 (v7g) | 0.98845 | 0.98911 |
| same, one stage 3 on mean logits (v7g2) | 0.98844 | 0.98911 |
| e5-small + bge, blend 0.3 / 0.7 (v7p) | 0.98878 | 0.98950 |
| bge + e5-large, 0.7 / 0.3 | - | 0.98954 |
| second pipeline (bge 3 folds) | 0.99112 | 0.99183 |
| v10a blend 0.6 ours / 0.4 second | - | 0.99205 |
| v10d stacker | - | 0.99226 |

Sources: [E: Transformer versions; 3-fold transformers; e5-large; v9 transformer stage; Second final-upload hunt].

#### B.3 Blocking audit

100k labelled Latin-name records per country whose true S1 row is in the held-out half, searched against all train S1 rows of the country [E: Blocking and cleaning audit].

| Step (added one at a time) | US recall | India recall | Candidates per record (US / India) |
|---|---:|---:|---|
| production search | 0.9868 | 0.9784 | 31.4 / 30.3 |
| + name search cap 4,000 -> 20,000 | 0.9910 | 0.9836 | 31.5 / 30.4 |
| + word searches min_df 2 -> 1 | 0.9921 | 0.9852 | 31.4 / 30.2 |
| + no-address name top 30 -> 100 | 0.9943 | 0.9884 | 34.5 / 33.3 |
| + fine-tuned e5 top 10 for every record | 0.9971 | 0.9956 | 42.0 / 41.4 |
| (production + e5 top 10 only) | 0.9941 | 0.9916 | 39.3 / 38.6 |

- **Why the name search was starved.** At cap 4,000 a median US name kept 4 character 3-grams in the search, and 3.2% kept none ("thomastechnologies": every 3-gram is in more than 4,000 US S1 names). Exact unique names were found 86% of the time; at cap 20,000, 99.98%.
- **Cost of the name cap** (union recall at the same candidate count):
  - cap 10,000: US +0.0033 / India +0.0032, at 43 s / 25 s per 100k queries;
  - cap 20,000: +0.0042 / +0.0052, at 124 s / 65 s;
  - production: 6 s.
- **Other variants, one at a time:**
  - BM25 instead of TF-IDF: +0.0000 US / +0.0009 India;
  - name + address top 30 / 40: +0.0018 / +0.0029 US at +9 / +19 candidates;
  - S1 initials added to the documents: India -0.0010;
  - without the learned abbreviation map: US -0.0019.
- **Dense search alone (e5 top 20):** 0.9876 US, above the whole TF-IDF union with 20 instead of 31 candidates.
  - FAISS IVFFlat is below exact search at useful speeds (US recall@20 0.873 at nprobe 8, 0.920 at 32).
  - Exact GPU search takes about 1 ms per query against 1.32M US S1 rows.
- **Cleaning fixes:**
  - India state codes: address token-set of affected true pairs 91.2 -> 97.5;
  - ordinals: affected US records' recall 0.9352 -> 0.9834.

#### B.4 Transformer as the final matcher

[E: Deep learning as the final matcher]
- Sample: 80,218 held-out S1 rows (US, India) with all their records and decoys at the same rate. 15.07M candidate pairs from the audited search.
- Each method sees each record's same top-5 candidates and the same decision-rule search.
- The sample has fewer competing decoys than the full data, so only comparisons within this table hold.

| Final matcher | All | US | India |
|---|---:|---:|---:|
| GBDT, stage 1 only (57 features, XGBoost depth 10) | 0.98998 | 0.99042 | 0.98954 |
| fine-tuned bge-reranker-v2-m3 alone | 0.99321 | 0.99308 | 0.99336 |
| stack of both, all records | 0.99362 | 0.99335 | 0.99389 |
| stack on close calls only (34% of records), GBDT elsewhere | 0.99351 | 0.99327 | 0.99378 |

- The transformer is the better single matcher (+0.0032).
- The close-call design keeps almost all of the stack's gain (0.99351 vs 0.99362), which is why the final pipeline scores close calls only.

#### B.5 Tried and not used

| Idea | Result | Source |
|---|---|---|
| Training on resampled "test-like" data (pairs of removed S1 rows deleted) | LB -0.0086 (v4): candidate lists halved, which the real search at lower density does not do | [E: v4 LB] |
| Stage 1 as a deeper XGBoost (depth 10, 1,000 rounds) | 0.98005 vs 0.98015 | [L: v4a] |
| Self-training on test pseudo-labels (1.03M test rows) | 0.97921 vs 0.97926 without | [E: v6 ingredients] |
| Isotonic calibration, per-country rules | +0.0002 held-out; adds about 56k borderline test matches | [E: v6 on FULL data] |
| Per-segment thresholds | +0.0000 | [E: v2 pipeline on S10] |
| Blend of two GBDT bases (v7blend) | 0.98801 vs 0.98805 | [E: Transformer versions] |
| Stage 3 with segment features | 0.98806 vs 0.98805 | [E: Transformer versions] |
| One stage 3 over the mean logits of two families | 0.98844 vs 0.98845 | [E: 3-fold transformers] |
| e5-large in a fixed blend | 0.98946 (three families) vs 0.98950 | [E: e5-large] |
| Widening the close-call band | only 1,425 held-out misses lie outside it | [E: Remaining held-out misses] |
| ID proximity, file order, records per source, for no-address chain records | no signal beyond chance, except the raw name (42.3% vs 26.2%), too weak under F0.5 | [E: Shortcut checks] |
| French address cleaning; France transformer additions; France number-changed additions; France restores; per-pattern France recalibration; France house-number veto | refuted (section 4) | [E], [L] |

#### B.6 Compute and hardware

| Machine | Hardware | What ran there |
|---|---|---|
| Laptop | Intel i5-12450H, 16 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11 | cleaning, blocking, stages 1-3, e5-small and e5-base cross-encoders, France rules, stacker, assembly |
| Workstation (Bhavya Jain) | NVIDIA RTX A6000 48 GB | bge-reranker-v2-m3 3 fold models; e5-large folds 0-1; bge scores of the wide-search pairs |
| Lab GPU server (Gathik Jindal, shared) | NVIDIA RTX PRO 6000 Blackwell 96 GB | second pipeline; e5-large fold 2 |
| Laptop (Harsh Gupta) | NVIDIA RTX 4060 8 GB | e5-base 3-fold run of v7g (not in the final file) |

Laptop wall-clock per step [M: `compute_cost.py`, parsed from `work/*.log`]:

| Step | Wall-clock |
|---|---:|
| fine-tune multilingual-e5-small for the embedding search (276,117 pairs) | 887 s |
| search + 55 features, all train (319.0M pairs) | 2.41 h |
| stage 1 (3 folds + full model) and a LightGBM stage 2 / XGBoost stage 2 with the group features | 1.68 h / 780 s |
| search + features, test (307.2M pairs) | 2.20 h |
| stage 1 + stage 2 on every test pair | 1.30 h |
| ranks 3-5, train / test | 1,102 s / 1,614 s |
| France cross-encoders (e5-small A, B; e5-base A2; scoring; stage 3; blend), incl. restarts | 8.2 h |
| e5-small, 3 fold models, training + scoring | 6.2 h |
| wide search + features, US/India test (424.9M pairs) | 3.63 h |
| stage 1 + ranks 3-5 + stage 2 on the wide pairs (469.8M) | 1.54 h |
| stage 3 for both families, blend, decision | about 14 min |

- The laptop steps above sum to 28.8 h (estimate of total laptop compute; it includes restarts and leaves out unlogged interactive analyses).
- bge-reranker-v2-m3 took about 3.4 h per fold (training + scoring) on the RTX A6000, partly shared with another user's job.
- The second pipeline's transformer stage took 2.45 h on the server (about 27 min training and 20 min scoring per fold). The full server run time was not recorded.
- Each of the stacker's 4 LightGBM models trained on about 3.0M pairs in 20-42 s (`src/logs/stack_fit_rob.log`, `stack_fit_robtr.log`).

Throughput [M: `compute_cost.py`]:
- e5-small training 359-384 pairs/s and scoring 1,946-1,979 pairs/s on the RTX 3050.
- bge-reranker-v2-m3 scoring 1,898 pairs/s on the RTX A6000.
- Disk: `work/` holds 98.5 GB, of which pair features are 63.6 GB.
- Every laptop step fits in 16 GB of RAM when steps run one at a time.

#### B.7 Models and licences

Every pretrained model the code loads. Licences are the `license` tags of the models' Hugging Face Hub pages, read on 27 Sep 2026 [R: code README section 10].

| Model | Licence | Parameters | Use in the final file |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M (counted from the weights) | embedding search (fine-tuned); cross-encoders (US/India 3 folds; France halves models) |
| intfloat/multilingual-e5-base | MIT | 278.0M (counted) | cross-encoder A2 (France rows) |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M (counted; the Hub page gives the same 567,755,777) | cross-encoder, 3 folds, in both pipelines |
| intfloat/multilingual-e5-large | MIT | 559.9M (computed from its published configuration; weights not on the laptop) | cross-encoder, 3 folds; stacker input |
| microsoft/mdeberta-v3-base | MIT | about 276M (published figure) | none: only the default of `rerank.py train` when no model is named; every documented command names one |
| LightGBM 4.7.0 | MIT | tree ensembles | stage 1, stacker |
| XGBoost 2.0.3 | Apache-2.0 | tree ensembles | stages 2 and 3; the second pipeline's stage 1 |

- **Counted:** the sum of tensor sizes in `model.safetensors`, read from the file header on 27 Sep 2026, for the Hub checkpoints and for the saved fine-tuned models (`work/e5_ft_addr/`, `work/ce/model*/`).
- **e5-large:** its published configuration is XLM-RoBERTa large, the architecture of bge-reranker-v2-m3, with 514 positions instead of 8,194. The layer-by-layer formula reproduces bge's counted 567,755,777 exactly and gives 559,890,432 for e5-large (559,891,457 as a cross-encoder). The Hub page rounds this to 560M.
- **Size limit:** every model is under 0.6B parameters, against the 8B limit, and carries an MIT or Apache-2.0 licence. Summed over the 16 fine-tuned transformer copies behind the final file, the total is 6.07B:
  - e5-small: 6 copies (the embedding model, A, B and 3 folds);
  - e5-base: 1 (A2);
  - bge: 6 (3 folds per pipeline);
  - e5-large: 3 folds.
- **Libraries:** every package in `requirements.txt` has an MIT, Apache-2.0 or BSD licence in its installed metadata. torch, numpy, scipy, scikit-learn and protobuf are BSD. The table with versions is in code README section 10.

#### B.8 Fair play and data use

**External data and network access** [R: code README section 10]
- **No external data, APIs, gazetteers, geocoding or lookup services.** Nothing looks up a business or an address outside the provided files.
- On 27 Sep 2026 we read every import in the 334 Python files and all 18 shell scripts of `code/business_entity_resolution/`. None imports a network library (requests, urllib, http, socket or similar), and none holds an API key or other credential. On 28 Sep 2026 the import scan was repeated over all 335 Python files, including the new `same_address.py`, with the same result.
- The only network access:
  - `from_pretrained` downloads the public checkpoints of B.7 on first use;
  - `pip install` fetches the packages at setup;
  - unattended scripts in `src/runners/` moved our own intermediate files between the team's machines by SSH/SFTP (the helper is not included) and pushed submission records to the team's git repository. Reproduction needs neither.

**Training data**
- Every learned table and model is fitted on the provided training data only:
  - the address maps of `learn_maps.py` (short forms and local-script state names, from train matches);
  - all tree models, including the stacker;
  - the fine-tuning of the transformers, which start from the public checkpoints of B.7.
- **Hand-written word lists** hold legal forms, titles, noise and descriptor words, French street types and region names, India state codes, US and India state abbreviations and spelled-out ordinals. They are in `normalize.py`, `finalize.py`, `same_address.py` (street types, filler and unit words of the address key) and `france_fix/artifacts/census/ops.py`.
  - France has no labels. The French entries were chosen by reading France test records and counting the word differences between them and the S1 rows they confidently match (`normalize.py`, `france_fix/artifacts/census/`).
- The 25 Sep forum rules noted in `EXPERIMENTS.md` allow hand-written normalisation dictionaries, unsupervised statistics on test and self-training. They forbid libpostal, gazetteers and APIs.

**Country as an open set** [R: code README section 0]
- The pipeline code (`src/*.py`, `src/stack/`, `src/movers/`) names no country and holds no fixed list of countries. `common.py` reads the countries from the `country` column of the raw Source 1 files.
- The countries with training labels are those of `train_source1.tsv` (US, India). The US/India steps (wide search, list movers, the second-pipeline blend, the stacker) apply to them.
- The countries without training labels are the other countries of `test_source1.tsv` (France). The rules chosen without labels (the min rule, legal-form veto and second-pipeline recall of section 4; the same-address rule of section 3) apply to them.
- Three narrower sets are also read from the data:
  - the countries whose records contain non-Latin-script names (India): used by the embedding model's recall check and by an optional stage-3 feature that the final file does not use;
  - the countries whose S1 addresses name the states Tamil Nadu, Delhi or Orissa in more rows than they carry the short forms TN, DL, OD or Odisha (India: 382,283 rows against 656; US: 190 against 89,794): used by the second pipeline's cleaning;
  - the stacker's country feature, the country's position in the list of countries with training labels (US 0, India 1).
- The France pair sets in `sets/` are fixed lists of France test pairs, chosen by the France analyses of section 4. The other sets are the US/India list movers (written by `movers/build_movers.py`) and the two stacker check sets.
- One file outside those folders is imported by the pipeline: the change detector `src/france_fix/artifacts/census/ops.py` (steps 25-26 of the code README). Its optional `country` argument compares the label with "France" and "India"; only the analysis scripts pass it, and the pipeline (`france_recall.py`, `same_address.py`) calls the detector without a country, so no pipeline output depends on it.
- Replacing the country names in the code changed no output: on 28 Sep 2026 `build_final.py` rebuilt v10a-v10d with the same md5 values as before (Appendix A), and each helper returned exactly the set of countries the code used to name.

**Test records** (the test set has no labels) were used in three ways:
1. **Inference.** Some features are statistics over the test files themselves, computed without labels:
   - the TF-IDF document frequencies of each search index are fitted on that country's test S1 rows;
   - the chain size counts the test S1 rows that share a core name;
   - the stage-2 group features compare the test records that point at the same S1 row.
2. **Unsupervised statistics for designing and checking rules:**
   - the comparison of test with train in section 2.1 (records per S1 row, look-alike share, density);
   - for France, the label-free tests on France test records: the all-lowercase share, the house-number direction and the change classes of `ops.py`. These chose and checked every France pair set (section 4; `code/business_entity_resolution/sets/README.md`).
3. **Tried and not used:**
   - training on "test-like" train data, resampled to match test's density and look-alike share (v4);
   - self-training on 1.03M pseudo-labelled test rows (B.5).

**The public leaderboard** returned one score per upload; 17 uploads have a recorded score [L]. The scores were used in four ways:
- **Choosing versions.** For example, v4's test-like training and v6's French cleaning were dropped after their scores.
- **Probe uploads** (section 5):
  - v5 and v7ens_frempty emptied every France row, which splits the score into a France part and a US/India part;
  - v7i, v7j, v7m and v10b changed only France rows, so each measured one France rule;
  - v9x put the second pipeline's US/India rows next to our France rows.
- **Fitting one France model.** A logistic model of France pair truth rates (`src/france_fix/lbcal/`, reused by `france_fix/build/b10_classmodel.py` and `b11_free.py`) was fitted to France values derived from the public scores of v7ens_frempty, v7i, v7j, v7m and v7g_num.
  - The fit wrote no pair set.
  - It gave second estimates of true shares, next to the lowercase test: 0.77-0.81 for the 9,800 noise-word additions, and 0.75-0.8 for the 4,150 earlier missed-match pairs [L: v9b, v9f].
- **Reading bundled results.** The v9zm upload, which carried the v9z France rows, showed France noise-suffix pairs to be about 90% true. On that evidence a 13,092-pair France veto was rejected (section 4, "Rules the leaderboard or the labels refuted").

No identifier, row order or train/test overlap signal is used (section 2.1, "No shortcuts").
