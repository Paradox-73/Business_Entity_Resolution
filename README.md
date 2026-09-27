# Business Entity Resolution
### Amazon ML Challenge 2026 · Team Mommy's Good Boys

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2E8B57?style=flat-square) ![XGBoost](https://img.shields.io/badge/XGBoost-GPU-FF6600?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Public LB](https://img.shields.io/badge/Public%20LB-0.990565-0F9D58?style=flat-square) ![Candidates](https://img.shields.io/badge/Candidates-7.07%20per%20entity-6E56CF?style=flat-square) ![Largest model](https://img.shields.io/badge/Largest%20model-568M%20params-4169E1?style=flat-square)

Given **1.73M business entities** (Source 1) and **9.97M noisy records** from two other sources (Source 2 and
Source 3) across the US, India and France, find every record that describes the same business. Each record is
searched only against its own country's entities, and the final candidate file keeps **7.07 candidates per entity**.
A gradient-boosted cascade compares each record with the other records that claim the same entity. Fine-tuned
cross-encoders re-read the uncertain 23.5% of records. For the US and India, a LightGBM stacker combines these scores
with those of a second run of the pipeline. An expected-F0.5 rule then chooses each entity's final match set.

> **Final public leaderboard score: 0.990565** (version `v10d`, the last upload, 27 Sep 2026).
> Held-out macro F0.5 on labelled US/India data: **0.99226**. France is 15% of the test set and has **no training
> labels**. France rules were checked with label-free tests built from quirks of the program that generated the data,
> with the analogous pairs in labelled US/India data, and, where possible, with uploads that changed only France rows.
> Every model has at most **568M parameters** (limit 8B) and an MIT or Apache-2.0 licence; **no external data, APIs or
> lookups** are used.

---

## Team

| Member | GitHub |
|---|---|
| Kanav Bhardwaj | [@Paradox-73](https://github.com/Paradox-73) |
| Bhavya Jain | [@Bhavzzzzzz](https://github.com/Bhavzzzzzz) |
| Harsh Gupta | [@Reverent2005](https://github.com/Reverent2005) |
| Gathik Jindal | [@gathik-jindal](https://github.com/gathik-jindal) |

---

## The task in numbers

| | Train | Test |
|---|---|---|
| Source 1 entities (S1) | 2,206,821 (US, India) | 1,732,544 (US 663k, India 810k, **France 259k**) |
| Source 2 + Source 3 records | 10,320,219 | 9,969,589 |
| Records per S1 entity | 4.68 | 5.75 (more look-alikes on test) |
| True matches | 7,638,365 pairs, 3.46 per entity; 5.6% of entities have none | hidden |

- **Metric:** F0.5 per S1 entity, averaged over entities. F0.5 weights precision twice as much as recall. An empty
  truth scores 1 only for an empty prediction.
- **What makes it hard:**
  - 38% of S1 rows share their exact name with another S1 row of the same country ("Primary Care Group" appears 253
    times).
  - 26% of records match nothing (decoys). Many imitate an S1 row: the first house number goes up by 1-20 for 68% of US
    decoys and 35% of India decoys (against their best candidate), and the name often gains or swaps a word.
- **The leaderboard is about 0.85 × US/India + 0.15 × France,** because France is 15.0% of the test S1 rows and every
  row counts equally (assuming the public subset has the same country mix).
- Train figures: `code/business_entity_resolution/src/experiments/measure/data_facts.py` (methodology section 2.1).

### Terms

| Term | Meaning |
|---|---|
| S1 row | one Source 1 entity; record = one Source 2 or Source 3 row |
| decoy, look-alike | a record that matches no S1 row, often made to resemble one |
| GBDT | gradient-boosted decision trees (LightGBM or XGBoost) |
| p1, p2 | probability that a pair is a true match, from stage 1 and from stage 2 |
| close call | a record whose best p2 is in [0.01, 0.995], or whose 2nd candidate has p1 >= 0.2; the transformers rescore these |
| cross-encoder, family | a transformer that reads both records' text together and scores the pair; a family is one such model type trained as 3 fold models |
| fold model, out-of-fold | fold model k trains on the labelled close calls of the other two folds; an out-of-fold score comes from a model that did not train on that row |
| held-out | macro F0.5 on the evaluation half of the train S1 rows (below) |
| second pipeline | the same code run a second time with wider blocking on a lab GPU server, with its own models |
| list-mover fix | where the wide and the old US/India candidate lists disagree on a pair, a transformer-only probability decides |
| France rule names | min rule, legal-form veto, descriptor veto, noise-word additions, false-accept veto, recoveries: see the table under "France without labels" |

---

## Results

Each upload tested one change where possible. The full log of 53 built versions is in
[`submissions/LOG.md`](submissions/LOG.md).

| Version | What changed | Held-out (US/India) | Public LB |
|---|---|---|---|
| v2 | Word-level blocking validated at full density, two-stage LightGBM, expected-F0.5 rule | 0.97904 | 0.9677 |
| v3 | + sibling features, XGBoost stage 2 | 0.98015 | 0.9700 |
| v7ens | + cross-encoders on close calls (e5-small, e5-base); France: min rule + legal-form veto | 0.98813 ¹ | 0.983159 |
| v9y | 3-fold bge-reranker + e5-small, wide US/India search, second pipeline's pairs; France: descriptor veto, noise-word additions, false-accept veto | 0.98950 | 0.989563 |
| v9zm | + 5,883 France recoveries, US/India candidate-list fix | same | 0.989968 |
| v10a | US/India probability = 0.6 × ours + 0.4 × the second pipeline's | 0.99205 | 0.990166 |
| v10b | + 4,825 France matches from the second pipeline's wider search | same | 0.990475 |
| v10c | + 802 France pairs (typos, same-address acronyms, '&' written 'et') | same | not uploaded |
| **v10d** | US/India LightGBM stacker over both pipelines; − 256 France pairs | **0.99226** | **0.990565** (final) |

- Held-out = macro F0.5 on the evaluation half of the train entities, searched against all train entities of the
  country (a full-size index like test's, not a sample; the US train index has twice as many S1 rows as the US test
  index).
- From v9y on, the held-out number gives every close call its out-of-fold transformer score ("all close calls
  rescored").
- ¹ v7ens uses the older "halves" protocol, in which some records keep the GBDT score. v9y scores 0.98878 on that
  protocol.
- The v10a blend gained +0.00019 held-out macro F0.5 on US/India (about +0.00016 on the leaderboard, since US/India
  is about 85% of it) and moved the public score by +0.000198.
- The v10d stacker gained +0.000209 on held-out. v10d moved the public score by +0.00009 over v10b, about 45% of the
  gain expected for the stacker and the France changes together (`submissions/LOG.md`).

**Probe uploads that measured France alone.** An upload with every France row emptied puts almost all of the score on
US/India; the difference to a full upload is 0.15 × (France score − 0.056), where 0.056 is the share of S1 rows with no
true match (train's rate, assumed for France), which score 1 when left empty.

| Probe | Public LB | What it showed |
|---|---|---|
| v5 (France emptied) | 0.838 | France ≈ 0.937, US/India ≈ 0.976 |
| v7ens_frempty | 0.848792 | France ≈ 0.952, US/India ≈ 0.9887: France was ~70% of the gap to the top |
| v7i (transformer may add France matches) | 0.982636 | France −0.0035: additions were net wrong, the min rule stays |
| v7m (restore same-address France pairs) | 0.9818 | France −0.0091: 16.7k of them were descriptor swaps ("Club" → "Ecole"), the start of the France fix |

---

## How it works

```
  S2/S3 record ──► cleaning (maps learned from train matches only)
                          │
                          ▼
       4 top-k searches against the S1 rows of the SAME country
       name char 3-grams · address words · name+address words · fine-tuned e5 embedding
                          │   ~31 candidates per record (~50 for US/India with the wide search)
                          ▼
       stage 1   LightGBM, 55 pair features ──► keep top 1-2 per record (+ ranks 3-5 of close calls)
                          │
                          ▼
       stage 2   XGBoost + group features: siblings, consensus, competition for the same S1
                          │
                          ▼
       close calls (23.5% of test records) ──► cross-encoders, 3 fold models per family
                          │                     e5-small (118M) · bge-reranker-v2-m3 (568M) · e5-large (~560M)
                          ▼
       stage 3   XGBoost per family ──► blend 0.3 × e5-small + 0.7 × bge
                          │
                          ▼
       US/India: LightGBM stacker over both pipelines  ◄──  second pipeline: same code, wider
                 (3 families, blend, GBDT scores,           blocking, own XGBoost + bge models
                  rank and margin in the record)
                          │
                          ▼
       expected-F0.5 set per S1        +        France rules and pair sets (no labels)
                          │
                          ▼
       matching_results.tsv            +        candidate_pairs.tsv (7.07 per S1)
```

**The second pipeline** is the same code run by Gathik on a lab GPU server with the audited, wider blocking
(`BER_V8=1`). Its search holds 99.63% of held-out true matches, against 98.51% for ours. It has its own models
(XGBoost stage 1, 3 bge-reranker folds) and contributes four things:
- matches for records our lists miss (v9y);
- a second per-pair probability, first in a fixed blend (v10a), then as stacker features (v10d);
- its pairs in the final candidate file, which the stacker scores;
- France matches for records we leave unmatched (v10b, v10c).

---

## Candidate generation

Every search runs **from each record towards the S1 rows of its own country** and returns a fixed number of
candidates, so no pair is compared exhaustively. Searches are sparse matrix products with a per-row top-k
(`sparse_dot_topn`), run in chunks. Terms shared by more than a fixed number of S1 rows are skipped during the
search, which bounds the work per record as the index grows.

| Search | Keys | Top-k | Term cap |
|---|---|---|---|
| Name | TF-IDF of character 3-grams of the cleaned core name | 10 (30 without address) | 4,000 S1 names |
| Address | TF-IDF of word 1-2-grams of the cleaned address | 10 | 5,000 S1 rows |
| Name + address | TF-IDF of word 1-2-grams of both | 20 (wide US/India test run: 40) | 5,000 (wide: 20,000) |
| Embedding | multilingual-e5-small fine-tuned on (record, S1) pairs, "name \| address"; records with non-Latin-script names only (every record in the second pipeline) | 10 | exact GPU search |

A stage-1 LightGBM model then keeps each record's best 1-2 candidates, plus candidates ranked 3-5 for close calls.
The pairs that the final models score form `candidate_pairs.tsv`.

| Measure | Value |
|---|---|
| Search recall, held-out at full density | 0.7968 (first version) → **0.9851** (production) → **0.9963** (audited, second pipeline) |
| Non-Latin names: embedding recall@10 against 883k India S1 rows | 0.156 off the shelf (name only) → 0.573 fine-tuned on names → **0.998** fine-tuned on name + address |
| Pairs returned by the four searches (test) | 469.8M (47.1 per record) |
| **`candidate_pairs.tsv` (the pairs the final models score)** | **12,249,116 = 7.07 per S1 entity** (US 6.55, India 7.21, France 7.96); 184 entities without candidates |
| Reduction ratio against all same-country pairs | 0.99999818 (12.25M of 6.72 × 10¹² pairs) |
| Recall ceiling of the final candidate definition, held-out US/India | **0.98936** at 5.35 / 5.68 candidates per S1 (US / India) |
| Search time (union, 50k US records against 1.32M S1 rows) | ~14 s |

What the candidate file is made of:

| Part | Pairs |
|---|---:|
| each record's top 1-2 stage-1 candidates (the stage-2 input) | 10,687,724 |
| + stage-1 ranks 3-5 of close-call records | 954,149 |
| + the second pipeline's US/India stage-2 pairs, where not already present (the stacker scores them) | 601,651 |
| + pairs restored by the US/India list-mover fix | 29 |
| + France pairs of the second pipeline's matching file that the final file matches, where not already present | 5,340 |
| + pairs of the France same-address rule (`code/business_entity_resolution/src/same_address.py`) that the final file matches, where not already present | 223 |
| **= `candidate_pairs.tsv`** | **12,249,116** |

- The candidate counts were measured on 27 Sep 2026 from `submissions/v10d/candidate_pairs.tsv`
  (`code/business_entity_resolution/src/experiments/measure/test_candidates.py`), the last two rows from the log of
  `make_candidates.py`, and the
  held-out recall ceiling from the saved out-of-fold score files of both pipelines (`heldout_recall.py`).
- The wide search exists for test only, so the held-out recall ceiling does not include it.
- All 5,862,410 matched pairs are candidates; the official validator checks this.

**Lesson learned:** a 10%-sample validation hid a collapse. Term caps tuned on the sample dropped every common 3-gram
once the index had full density, and search recall fell to 0.7968. Since then, every blocking number is measured
against the full index.

---

## Matching

| Stage | Model | Input | What it adds |
|---|---|---|---|
| 1 | LightGBM, 255 leaves, 600 rounds | 55 features: name and address similarities, house-number agreement, chain size, margin to the record's next-best candidate | ranks each record's candidates |
| 2 | XGBoost (GPU), 76 features | p1 + **group features**: the other records that point at the same S1, their agreement with this record, the same house number | look-alikes disagree with the entity's other records (+0.0025 held-out) |
| cross-encoders | 3 fold models per family, 1 epoch on 1.68M close-call pairs each | lowercased "name \| address" of both sides, 128 tokens | the transformer reads the raw text (+0.0070 held-out) |
| 3 | XGBoost per family | p1, p2, cross-encoder logit, its margin and rank | combines GBDT and transformer |
| stacker (US/India) | LightGBM, 15 leaves, 300 rounds; mean of 4 cross-fitted models | 24 per-pair features, among them: 3 family probabilities, the blend, the second pipeline's probability, both pipelines' GBDT p1/p2, which pipeline scored the pair, rank and margin in the record, country, no-address flags, house-number difference, exact name | +0.000209 held-out over the fixed 0.6 / 0.4 blend |
| decision | expected-F0.5 set selection | final probability | per S1: the prefix of records that maximises expected F0.5, or empty |

- Each fold model trains on the close calls of the other two folds, so every labelled close call has an out-of-fold
  score for stage 3.
- Held-out AUC (the chance that a random true pair scores above a random false pair) on the same close calls, records
  spanning both S1 halves left out: e5-small 0.9908–0.9914, bge-reranker-v2-m3 0.9937–0.9941 (METHODOLOGY B.1).
- The stacker leaves out three list-length features: the record's candidate count (`nq`) and the pair's rank and
  margin within the S1 row (`rank_s`, `margin_s`). Test has more candidates per S1 row than held-out (7.07 vs
  5.35-5.68).

---

## France without labels

The data comes from a simulator (a program that generated the records). Two quirks of that simulator, measured on
labelled US/India data (`code/business_entity_resolution/src/experiments/measure/`, METHODOLOGY section 2.1), became **label-free tests** for France:

- **Look-alikes move the house number up** (US decoys: +2 to +20 in 60.5% of cases, India 31.0%). True records rarely
  change it, and move it up or down about equally (US 0.8% up / 0.8% down, India 1.9% / 2.1%).
- **True records whose name adds or swaps in a noise or descriptor word are almost never all-lowercase** (US 0.03%,
  India 0.22%); decoys with the same kind of change are all-lowercase 3.01% (US) and 2.39% (India). For other changed
  words there is no difference (US true 2.84%, decoys 2.81%). So the lowercase share estimates the true share only of
  France pair sets defined by noise or descriptor words.

| France rule | Pairs | Evidence |
|---|---|---|
| Legal-form veto (SARL vs SAS, no form in common) | p = 0 | such conflicts are true 5.2% (US) / 0.0% (India) of the time |
| Min rule: transformers may only lower a France probability | all | letting them add France matches cost −0.0035 France (v7i) |
| Descriptor-word veto ("Club" ↔ "Ecole", "Amicale", "Comité" …) | 22,436 removed | lowercase test: ~0% true. In US these swaps are 98.9% true, in France they are look-alikes |
| Noise-word additions (same address, "& Fils", "Services" …) | 9,800 added | lowercase test ~0.96 true; a fit to the France probe results gave 0.77-0.81 |
| False-accept veto: legal form (846) or a word (232) added and house number up by 1-20, or an all-lowercase record with a name change (67) | 1,145 removed | the look-alike signature; estimated true share A 0.0, B 0.60, C 0.42 (column `t` of `sets/fp_veto_set.parquet`) |
| Recoveries | 5,883 added | estimated true share 0.93 on average (column `est` of `sets/recall_add_set.parquet`); v9zm (with a US/India list fix) +0.0004 public LB |
| Second pipeline's matches outside our lists (v10b) | 4,825 added | the same kind of pair is 98.8% true on US/India labels; France +0.00206 on the leaderboard |
| Garbled-word typos, same-address acronyms, '&' written 'et' or '+' (v10c) | 802 added | the subsets that survived a separate attempt to refute each set; US/India analogs 99.3% true (typos), 99.85% (same-address rule) |
| Same generic name and house number, completely different street, both bge families < 0.3 (v10d) | 256 removed | the US/India analog is true 1 time in 87 |

France rose from ≈ 0.952 (v7ens) to ≈ 0.98 (v9y onward), an estimate from the leaderboard; v10b added +0.00206.

**In the code, country is an open set.** The pipeline code names no country and holds no fixed country list:
- `common.py` reads the countries from the `country` column of the raw Source 1 files. The countries with training
  labels are those of `train_source1.tsv` (US, India). The countries without them are the other countries of
  `test_source1.tsv` (France).
- The rules above apply to every test country without training labels; France is the only such country in this test
  set. The steps described as US/India apply to every country with training labels.
- The France pair sets in `sets/` are fixed lists of France test pairs, chosen by the analyses in `src/france_fix/`.
  One of them, the 298 same-address pairs, is also regenerated by the pipeline module `same_address.py`.
- Section 0 of the [code README](code/business_entity_resolution/README.md) lists the helpers and the three narrower
  sets that are also read from the data.

---

## What moved the score, and what did not

| Worked | Gain |
|---|---|
| Measuring blocking at full density, then fixing the caps | search recall 0.7968 → 0.9851 |
| Group features (siblings, consensus) | +0.0025 held-out; +0.0023 public LB for the sibling features alone |
| Cross-encoders on close calls only | +0.0070 held-out; scoring every pair gains almost nothing more (0.99351 vs 0.99362 in a test) |
| 3-fold transformers (2× the training data of two half-models) | AUC 0.9887 → 0.9912 (same model size) |
| France rules checked with the lowercase test | France ≈ 0.952 → ≈ 0.98 |
| Blend with the second pipeline | +0.000198 public LB (held-out +0.00019 on US/India) |
| France matches from the second pipeline's wider search (v10b) | +0.000309 public LB (France +0.00206) |
| LightGBM stacker over both pipelines (v10d) | +0.000209 held-out; +0.00009 public LB together with the v10c/v10d France changes |

| Did not work | Result |
|---|---|
| Training on "test-like" data made by deleting S1 rows' pairs | −0.0086 LB: it halved the US candidate lists (15.9 vs 31.3 per record), which the real search never does |
| French-specific address cleaning | LB fell (v6) |
| Self-training on test pseudo-labels | 0.97921 vs 0.97926 held-out |
| Letting transformers add France matches | −0.0035 France |
| Adding France records whose house number changed | −0.0163 France (v7j) |
| Picking the S1 row of a no-address chain record by closest id, most other records, or closest raw name | right 26.4% / 23.1% / 42.3% of the time vs 26.2% at random; too weak under F0.5 |
| multilingual-e5-large as a third family in the fixed blend | 0.98946 vs 0.98950 without it; kept only as a stacker feature |
| France pairs that only the second pipeline accepts, where both pipelines scored them | the US/India analog is 69.6% true; no set built |
| France vetoes where both bge families score low | 13,092 candidates refuted: 83% are 'X \<descriptor\> SARL' → '… Développement / Groupe / & Associés', which the v9zm leaderboard result (the upload that carried the v9z France rows) shows are ~90% true; only 256 kept (v10d) |
| Isotonic recalibration with per-country rules | +0.0002 held-out (0.98151 -> 0.98173); left out because it added about 56k borderline test matches in the band where test has twice train's look-alike share |

---

## Repository layout

| Path | What it is |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | **The solution.** Its [`README.md`](code/business_entity_resolution/README.md) lists every command, in order, from the raw data to both output files |
| `code/business_entity_resolution/src/` | Pipeline code: cleaning, blocking, stages 1–3, cross-encoders, rules for the countries without training labels (France), final assembly; `movers/`, `blend_second.py`, `france_recall.py`, `same_address.py`, `apply_pair_sets.py` and `stack/` for the versions after v9y |
| `code/business_entity_resolution/src/build_final.py` | One command that rebuilds v10a–v10d from the saved intermediate files and checks the md5 against the uploaded files |
| `code/business_entity_resolution/sets/` | The 13 pair sets and the stacker model file of the final build. [`sets/README.md`](code/business_entity_resolution/sets/README.md) lists rows, md5 and origin; the data files are in the zip, not in git |
| `code/business_entity_resolution/src/france_fix/` | The France analyses: census of the changes the data generator makes, look-alike and missed-match hunts, leaderboard fits, and the scripts that chose each France pair set ([`README.md`](code/business_entity_resolution/src/france_fix/README.md)) |
| `code/business_entity_resolution/src/runners/` | Scripts that ran on the lab GPU server and the unattended runs of 26–27 Sep |
| `code/business_entity_resolution/src/experiments/` | Experiments not needed for the final files; `measure/` holds the scripts behind the numbers the documentation measured on 27 Sep 2026 |
| [`tools/make_package.py`](tools/make_package.py) | Builds the Round 2 submission zip into `dist/` |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | **The lab notebook.** Every run, its held-out score, the decision taken, and the lessons |
| [`submissions/LOG.md`](submissions/LOG.md) | Every built version: what it tests, held-out score, public score |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Methodology write-up in the organisers' template; goes into the zip as `Documentation_template.md` |
| `docs/runbooks/` | How the transformer folds and the second pipeline were run on the teammates' GPU machines and the lab server |
| `2026_official/`, `student_resource/` | Official problem statement, rules, template and validator. The dataset itself is not in git |

`work/` (intermediate files, 98.5 GB on 27 Sep 2026), `output/` (the two final files) and `dist/` (the zip) are
created by the scripts and never committed.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
export BER_DATA=/path/to/student_resource/dataset         # train/ and test/ source files
export BER_WORK=/path/to/work                              # intermediate files (~100 GB)
export BER_OUT=/path/to/output                             # the two submission files
```

The main path, in order (the 28 steps with options, run times and machines are in the
[solution README](code/business_entity_resolution/README.md)):

```bash
python learn_maps.py && python prep.py                        # cleaning
python embed.py train && python embed.py encode               # embedding search for non-Latin names
python pipeline.py build train full                           # blocking + features, all of train
BER_BACKEND=lgb python pipeline.py train full                 # stage 1 (LightGBM); topk5.py reads its result.json
BER_BACKEND=lgb BER_BACKEND2=xgb python pipeline.py train full cons   # stage 2 (XGBoost, group features)
python pipeline.py build test test && python pipeline.py predict full_cons test
python rerank.py select full_cons                             # close calls
python topk5.py train full && python topk5.py test full && BER_CE_DIR=$BER_WORK/ce_x python rerank.py extras full_cons $BER_WORK/ce
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --no-freeze --batch 64 --lr 2e-5   # 48 GB GPU
python rerank.py stage3 && python blend.py ...                # stage 3, family blend
python finalize.py ... && python assemble_final.py ...        # decision rule, France rows: v9y
python movers/build_movers.py v7p v7q v9y $BER_WORK/movers    # US/India list movers (v9zm)
python blend_second.py ... && python france_recall.py ... && python apply_pair_sets.py ...   # v10a, v10b, v10c
cd stack && python build_ho.py && python fit.py gbdt ... && python build_test.py && python apply_test.py _avg   # v10d stacker
```

From the saved intermediate files, one command rebuilds v10a–v10d and compares them with the uploaded files:

```bash
python build_final.py
```

On 27 Sep 2026 it ran in 304 s on the laptop (`code/business_entity_resolution/src/logs/build_final.log`). The rebuilt
`matching_results.tsv` (md5
`18412329111b5d9c43df3a58df0574d1`) and `candidate_pairs.tsv` (md5 `a2ddcb5d26ede94f780fd2c4d87519a7`) were
byte-identical to the uploaded v10d files, and the official validator printed PASS. A second run on 28 Sep 2026,
after the country names were removed from the code and the same-address check was added, gave the same md5 values
(`code/business_entity_resolution/src/logs/build_final_28sep.log`).

Check a submission with the official validator:

```bash
python student_resource/utils/validate_submission.py -m output/matching_results.tsv \
    -c output/candidate_pairs.tsv -t student_resource/dataset/test --check-ids
```

**Where it ran**

| Machine | Hardware | What ran there |
|---|---|---|
| Team laptop | i5-12450H, 16 GB RAM, RTX 3050 4 GB | cleaning, blocking, stages 1–3, e5-small folds, France rules, the stacker, assembly |
| Bhavya's GPU machine | RTX A6000 48 GB | bge-reranker-v2-m3 fold models; e5-large folds 0-1 |
| Lab GPU server (Gathik) | RTX PRO 6000 Blackwell 96 GB, shared | the second pipeline; e5-large fold 2 |
| Harsh's laptop | RTX 4060 8 GB | multilingual-e5-base folds (earlier versions) |

---

## Round 2 submission package

The organisers ask for one zip, `<team_name>_submission.zip`, with `output/` (both files),
`code/business_entity_resolution/` (source under `src/`, `README.md`, `requirements.txt`) and the filled-in
`Documentation_template.md`. One command builds it:

```bash
python tools/make_package.py --validate      # -> dist/Mommys_Good_Boys_submission.zip
```

| Zip path | Taken from |
|---|---|
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | `output/` |
| `code/business_entity_resolution/` | `code/business_entity_resolution/`, including the `sets/` data files git ignores, without `__pycache__` and virtual environments |
| `code/business_entity_resolution/docs/` | copies of `EXPERIMENTS.md`, `submissions/LOG.md`, the `finalize.json` files of v9z-v10d and `docs/runbooks/`, which the READMEs and the methodology cite |
| `Documentation_template.md` | `docs/METHODOLOGY.md` |

- **Checks before writing:**
  - `output/` must hold the uploaded v10d files (md5); `--allow-other-output` skips this check;
  - the 14 files of `sets/` must exist with their md5;
  - `--validate` also runs the official validator.
- **Checks after writing:** the script reads the zip back and checks the CRC of every entry.
- **Files git does not track:** the script lists them, so the repository can be committed to match the zip.
- The team name is written without the apostrophe and spaces, to keep the file name portable; `--team` sets another.
- First test run on 27 Sep 2026 (laptop, 257 s including the validator): 395 files, 283.1 MB before compression,
  121.9 MB zipped.
- Final build on 28 Sep 2026 (318 s including the validator, which printed PASS): 397 files (335 Python files),
  283.2 MB before compression, 121,884,523 bytes zipped, zip md5 `46fa48289800c3659fca8d74be0e5b5c`. Its two TSV
  files are byte-identical to `submissions/v10d/`. The zip holds neither the dataset nor the 98.5 GB work folder.

---

## Models and licences

| Model | Licence | Parameters | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M | embedding search (fine-tuned); cross-encoder, 3 folds |
| intfloat/multilingual-e5-base | MIT | 278.0M | cross-encoder for the France rows |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M | cross-encoder, 3 folds (both pipelines) |
| intfloat/multilingual-e5-large | MIT | 559.9M | third cross-encoder family, 3 folds; its probability is a stacker feature |
| LightGBM / XGBoost | MIT / Apache-2.0 | tree ensembles | stages 1–3, the US/India stacker |

- The first three parameter counts are counted from the weight files (`model.safetensors`). The e5-large count is
  computed from its published configuration, because its weights stayed on the GPU machines; its Hub page rounds it
  to 560M.
- The 16 fine-tuned transformer copies behind the final file add up to 6.07B parameters, also under the 8B limit.
- `rerank.py train` names microsoft/mdeberta-v3-base (MIT) as its default model, but every documented command names
  another model, and mdeberta was not used.
- All learned tables and models are fitted on the provided training data only. The only network access is the
  download of the public checkpoints above and of the Python packages at setup (code README section 10).

---

## Limits, stated rather than found

- **Records without an address are the largest remaining error.** On held-out data they cost 0.0060 of the 0.0080
  US/India loss. When several S1 rows share the record's exact name, nothing in the record says which one it
  belongs to. Closest id, most other records, per-source coverage and closest raw name were tested; the best (raw
  name, right 42.3% vs 26.2% at random) is too weak to pay under F0.5.
- **France has no labels.** Its score is estimated from probe uploads and leaderboard differences (≈ 0.98 from v9y
  on), not measured. France rules rest on the label-free tests above, on US/India analogs and, for a few rules, on
  France-only uploads; the tests themselves are checked on US/India labels, not on France.
- **Held-out and leaderboard agree within a pipeline, not across pipelines.**
  - Changes made to our own pipeline moved the leaderboard in the direction held-out predicted; changes that handle
    decoys gained 1.6-2.1 times their held-out gain on the leaderboard (v3: +0.0011 -> +0.0023; v7ens: +0.0080 ->
    +0.0127 on US/India).
  - Held-out rated the second pipeline's US/India +0.0023 above ours, but the leaderboard put it 0.00035 below (v9x).
    We therefore decided blends by held-out gains within the same data, not by comparing absolute held-out scores.
  - v10d (stacker plus the v10c/v10d France changes) moved the public score by +0.00009, about 45% of the gain
    expected for both together; the two parts cannot be separated.
- **GPU training is not bit-for-bit deterministic.** A full rerun gives a file close to, not identical with, the
  submitted one. Every step from the saved scores on is exact: `build_final.py` rebuilds the final files byte for
  byte.
- **Some parts ship as saved outputs rather than re-runnable commands:**
  - the France pair sets, chosen by analysis scripts that ran once, interactively, in the scratch folder
    `$BER_SCRATCH`;
  - the stacker model file: the output logs of its two fit runs are in `src/logs/`, but the commands were
    reconstructed, not logged.
  - Both are in `sets/` with their md5.
- **The unattended-run scripts in `src/runners/` carry the laptop's and servers' folder names.** The helper that
  copied files between machines is not included because it holds login details.

---

## Where the depth is

| Read | For |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every experiment with its held-out score, error breakdowns, the blocking audit, and the lessons section |
| [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) | Reproducing both submission files, step by step, with machines and run times |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Problem analysis, blocking, features, models, error analysis |
| [`submissions/LOG.md`](submissions/LOG.md) | All versions and their leaderboard results |
| [`code/business_entity_resolution/sets/README.md`](code/business_entity_resolution/sets/README.md) | Every pair set of the final build and the scripts that chose it |
| [`code/business_entity_resolution/src/stack/README.md`](code/business_entity_resolution/src/stack/README.md) | The US/India stacker: features, validation, commands |
| `code/business_entity_resolution/src/france_fix/` | How each France rule was found and checked |
