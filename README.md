# Business Entity Resolution
### Amazon ML Challenge 2026 · Team Mommy's Good Boys

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2E8B57?style=flat-square) ![XGBoost](https://img.shields.io/badge/XGBoost-GPU-FF6600?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Public LB](https://img.shields.io/badge/Public%20LB-0.990166-0F9D58?style=flat-square) ![Candidates](https://img.shields.io/badge/Candidates-7.07%20per%20entity-6E56CF?style=flat-square) ![Largest model](https://img.shields.io/badge/Largest%20model-568M%20params-4169E1?style=flat-square)

Given **1.73M business entities** (Source 1) and **9.97M noisy records** from two other sources (Source 2 and
Source 3) across the US, India and France, find every record that describes the same business. Each record is
searched only against its own country's entities and keeps **7.07 candidates per entity**. A gradient-boosted
cascade compares each record with the other records that claim the same entity. Fine-tuned cross-encoders re-read
the uncertain 23.5% of records. An expected-F0.5 rule then chooses each entity's final match set.

> **Best public leaderboard score: 0.990166** (version `v10a`, 27 Sep 2026 17:46 IST, public rank 28).
> Held-out macro F0.5 on labelled US/India data: **0.99205**. France is 15% of the test set and has **no training
> labels**. Every France rule was checked with label-free tests built from quirks of the data generator. Every model
> has at most **568M parameters** (limit 8B) and an MIT or Apache-2.0 licence; **no external data, APIs or lookups**
> are used.

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
  - 39% of S1 names are shared with another S1 row ("Primary Care Group" appears 253 times).
  - 26% of records match nothing. They are look-alikes: the same street with the house number moved up a few units,
    and a word added or swapped in the name.
- **The leaderboard is 0.85 × US/India + 0.15 × France,** because France is 15.0% of the test S1 rows and every row
  counts equally.

---

## Results

Each upload tested one change where possible. The full log of 51 built versions is in
[`submissions/LOG.md`](submissions/LOG.md).

| Version | What changed | Held-out (US/India) | Public LB |
|---|---|---|---|
| v2 | Word-level blocking validated at full density, two-stage LightGBM, expected-F0.5 rule | 0.97904 | 0.9677 |
| v3 | + sibling features, XGBoost stage 2 | 0.98015 | 0.9700 |
| v7ens | + cross-encoders on close calls (e5-small, e5-base); France: min rule + legal-form veto | 0.98813 | 0.983159 |
| v9y | 3-fold bge-reranker + e5-small, wide US/India search, second generator's pairs; France: descriptor veto, noise-word additions, false-accept veto | 0.98950 | 0.989563 |
| v9zm | + 5,883 France recoveries, US/India candidate-list fix | same | 0.989968 |
| **v10a** | US/India probability = 0.6 × ours + 0.4 × the second pipeline's | **0.99205** | **0.990166** |
| v10b | + 4,825 France matches from the second generator's wider search | same | pending |

Held-out = macro F0.5 on the evaluation half of the train entities, searched against all train entities of the
country (the same density as test). The blend in v10a was predicted at +0.00019 on held-out and moved the public
score by +0.00020.

**Probe uploads that measured France alone.** An upload with every France row emptied puts all of the score on
US/India; the difference to a full upload is the France score × 0.15.

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
                          │   ~31 candidates per record (~50 with the wide US/India search)
                          ▼
       stage 1   LightGBM, 55 pair features ──► keep top 1-2 per record (+ ranks 3-5 of close calls)
                          │
                          ▼
       stage 2   XGBoost + group features: siblings, consensus, competition for the same S1
                          │
                          ▼
       close calls (23.5% of test records) ──► cross-encoders, 3 fold models per family
                          │                     e5-small (118M) · bge-reranker-v2-m3 (568M)
                          ▼
       stage 3   XGBoost per family ──► family blend ──► 0.6 × ours + 0.4 × second pipeline
                          │
                          ▼
       expected-F0.5 set per S1        +        France rules (no labels)
                          │
                          ▼
       matching_results.tsv            +        candidate_pairs.tsv (7.07 per S1)
```

**The second pipeline** is the same code run by Gathik on a lab GPU server with the audited, wider blocking
(`BER_V8=1`). Its candidate lists hold 99.63% of held-out true matches, against 98.51% for ours. It
has its own models (XGBoost stage 1, 3 bge-reranker folds). It contributes three things:
- matches for records our lists miss;
- a second per-pair probability for the blend;
- in v10b, France matches for records we leave unmatched.

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
| Embedding | multilingual-e5-small fine-tuned on (record, S1) pairs, "name \| address" | 10 | exact GPU search |

| Measure | Value |
|---|---|
| Shortlist recall, held-out at full density | 0.7968 (first version) → **0.9851** (production) → **0.9963** (audited, second pipeline) |
| Non-Latin names: embedding recall@10 against 883k India S1 rows | 0.156 off the shelf → **0.998** fine-tuned on name + address |
| Pairs after the four searches (test) | 469.8M (47.1 per record) |
| **`candidate_pairs.tsv` (the pairs the final models score)** | **12.25M = 7.07 per S1 entity**, 0.0002% of all same-country pairs |
| Search time (union, 50k US records against 1.32M S1 rows) | ~14 s |

**Lesson learned:** a 10%-sample validation hid a collapse. Term caps tuned on the sample dropped every common 3-gram
once the index had full density, and shortlist recall fell to 0.7968. Since then, every blocking number is measured
against the full index.

---

## Matching

| Stage | Model | Input | What it adds |
|---|---|---|---|
| 1 | LightGBM, 255 leaves, 600 rounds | 55 features: name and address similarities, house-number agreement, chain size, margin to the record's next-best candidate | ranks each record's candidates |
| 2 | XGBoost (GPU), 76 features | p1 + **group features**: the other records that point at the same S1, their agreement with this record, the same house number | look-alikes disagree with the entity's other records (+0.0025 held-out) |
| cross-encoders | 3 fold models per family, 1 epoch on 1.68M close-call pairs each | lowercased "name \| address" of both sides, 128 tokens | the transformer reads the raw text (+0.0070 held-out) |
| 3 | XGBoost per family | p1, p2, cross-encoder logit, its margin and rank | combines GBDT and transformer |
| decision | expected-F0.5 set selection | final probability | per S1: the prefix of records that maximises expected F0.5, or empty |

Each fold model trains on the close calls of the other two folds, so every labelled close call has an out-of-fold
score for stage 3. Held-out AUC on the same close calls: e5-small 0.9908–0.9914, bge-reranker-v2-m3 0.9937–0.9941.

---

## France without labels

The data comes from a simulator with fixed settings: US and India agree to 3–4 decimals on every count we measured. Two quirks
of that simulator, measured on labelled US/India data, became **label-free tests** for France:

- **Look-alikes move the house number up** (US decoys: +2..+20 in 59% of cases). True records move it up or down
  about equally.
- **True records with a changed word are almost never all-lowercase** (0.2–0.4%); look-alikes are lowercase at the
  normal rate (2–4%). The lowercase share of any France pair set therefore estimates how many of its pairs are true.

| France rule | Pairs | Evidence |
|---|---|---|
| Legal-form veto (SARL vs SAS, no form in common) | p = 0 | such conflicts are true 5.2% (US) / 0.0% (India) of the time |
| Min rule: transformers may only lower a France probability | all | letting them add France matches cost −0.0035 France (v7i) |
| Descriptor-word veto ("Club" ↔ "Ecole", "Amicale", "Comité" …) | 22,436 removed | lowercase test: ~0% true. In US these swaps are 98.9% true, in France they are look-alikes |
| Noise-word additions (same address, "& Fils", "Services" …) | 9,800 added | lowercase test ~0.96 true |
| False-accept veto (legal form added + house number up) | 1,145 removed | the look-alike signature |
| Recoveries | 5,883 added | lowercase test ~0.9 true; v9zm (with a US/India list fix) +0.0004 public LB |
| Second generator's matches outside our lists (v10b) | 4,825 added | the same kind of pair is 98.8% true on US/India labels |

France rose from ≈ 0.952 (v7ens) to ≈ 0.978–0.98 (v9y onward), estimated from the leaderboard.

---

## What moved the score, and what did not

| Worked | Gain |
|---|---|
| Measuring blocking at full density, then fixing the caps | shortlist recall 0.7968 → 0.9851 |
| Group features (siblings, consensus) | +0.0025 held-out; +0.0023 public LB for the sibling features alone |
| Cross-encoders on close calls only | +0.0070 held-out; scoring every pair gains almost nothing more (0.99351 vs 0.99362 in a test) |
| 3-fold transformers (2× the training data of two half-models) | AUC 0.9887 → 0.9912 (same model size) |
| France rules checked with the lowercase test | France ≈ 0.952 → ≈ 0.978 |
| Blend with the second pipeline | +0.0002 public LB, exactly as held-out predicted |

| Did not work | Result |
|---|---|
| Training on "test-like" data made by deleting S1 rows' pairs | −0.0086 LB: it halved the candidate lists, which the real search never does |
| French-specific address cleaning | LB fell (v6) |
| Self-training on test pseudo-labels | 0.97921 vs 0.97926 held-out |
| Letting transformers add France matches | −0.0035 France |
| Adding France records whose house number changed | −0.0163 France (v7j) |
| Rules for records without an address (record counts, per-source counts, raw and sibling names, a 32-feature re-scorer) | best +0.00003 held-out, i.e. noise |
| A third cross-encoder family, multilingual-e5-large | three-family blend 0.98946 vs 0.98950 without it; not used |
| Decision-rule variants: exact expected F0.5, recalibration, empty-set handling | all within ±0.0001 |

---

## Repository layout

| Path | What it is |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | **The solution.** Its [`README.md`](code/business_entity_resolution/README.md) lists every command, in order, from the raw data to both output files |
| `code/business_entity_resolution/src/` | Pipeline code: cleaning, blocking, stages 1–3, cross-encoders, France rules, final assembly |
| `code/business_entity_resolution/france_fix/` | The France analyses: generator census, look-alike and missed-match hunts, leaderboard fits |
| `code/business_entity_resolution/server/`, `overnight/` | Scripts that ran on the lab GPU server and the unattended runs of 27 Sep |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | **The lab notebook.** Every run, its held-out score, the decision taken, and the lessons |
| [`submissions/LOG.md`](submissions/LOG.md) | Every built version: what it tests, held-out score, public score |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Methodology write-up in the organisers' template (draft for Round 2) |
| [`docs/STRATEGY.md`](docs/STRATEGY.md) | The plan, roles and timeline written during the round |
| `docs/runbooks/` | Instructions for the teammates' GPU machines and the lab server |
| `docs/archive/2025_prep/` | Preparation written before the 2026 task was announced |
| `2026_official/`, `student_resource/` | Official problem statement, rules, template and validator. The dataset itself is not in git |

`work/` (≈90 GB of intermediate files) and `output/` are created by the pipeline and never committed.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
export BER_DATA=/path/to/student_resource/dataset         # train/ and test/ source files
export BER_WORK=/path/to/work                              # intermediate files (~90 GB)
export BER_OUT=/path/to/output                             # the two submission files
```

The main path, in order (the 20 steps with options, run times and machines are in the
[solution README](code/business_entity_resolution/README.md)):

```bash
python learn_maps.py && python prep.py                        # cleaning
python embed.py train && python embed.py encode               # embedding search for non-Latin names
python pipeline.py build train full                           # blocking + features, all of train
BER_BACKEND=lgb BER_BACKEND2=xgb python pipeline.py train full cons   # stages 1 and 2
python pipeline.py build test test && python pipeline.py predict full_cons test
python rerank.py select full_cons                             # close calls
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --no-freeze --batch 64 --lr 2e-5   # 48 GB GPU
python rerank.py stage3 && python blend.py ...                # stage 3, family blend
python finalize.py ... && python assemble_final.py ...        # decision rule, France rows, final file
python make_candidates.py ...                                 # candidate_pairs.tsv
```

Check a submission with the official validator:

```bash
python student_resource/utils/validate_submission.py -m output/matching_results.tsv \
    -c output/candidate_pairs.tsv -t student_resource/dataset/test --check-ids
```

**Where it ran**

| Machine | Hardware | What ran there |
|---|---|---|
| Team laptop | i5-12450H, 16 GB RAM, RTX 3050 4 GB | cleaning, blocking, stages 1–3, e5-small folds, France rules, assembly |
| Bhavya's workstation | RTX A6000 48 GB | bge-reranker-v2-m3 fold models; e5-large folds 0-1 |
| Lab GPU server (Gathik) | RTX PRO 6000 Blackwell 96 GB, shared | the second pipeline; e5-large fold 2 |
| A teammate's laptop | RTX 4060 8 GB | multilingual-e5-base folds (earlier versions) |

---

## Models and licences

| Model | Licence | Parameters | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M | embedding search (fine-tuned); cross-encoder, 3 folds |
| intfloat/multilingual-e5-base | MIT | 278.0M | cross-encoder for the France rows |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M | cross-encoder, 3 folds (both pipelines) |
| intfloat/multilingual-e5-large | MIT | ~560M | third cross-encoder family, tested and not used (no held-out gain) |
| LightGBM / XGBoost | MIT / Apache-2.0 | tree ensembles | stages 1–3 |

Parameter counts are from the saved weight files. All learned tables and models are fitted on the provided training
data only.

---

## Limits, stated rather than found

- **Records without an address are the largest remaining error.** On held-out data they cost 0.0060 of the 0.0080
  US/India loss. When several S1 rows share the record's exact name, nothing in the record says which one it
  belongs to. Record counts, per-source counts, raw and sibling names and a 32-feature re-scorer were tested; none
  beat abstaining.
- **France has no labels.** Its score is estimated from probe uploads (≈ 0.978–0.98 now vs ≈ 0.992 for US/India).
  Every France rule rests on the label-free tests above, which are checked on US/India labels, not on France.
- **Held-out and leaderboard agree within a pipeline, not across pipelines.** Changes made to our own pipeline moved
  the leaderboard as held-out predicted. But held-out rated the second pipeline's US/India +0.0023 above ours, while
  the leaderboard put it 0.00035 below (v9x). We therefore decide blends by held-out gains within the same data, not
  by comparing absolute held-out scores.
- **GPU training is not bit-for-bit deterministic.** A full rerun gives a file close to, not identical with, the
  submitted one. The final assembly steps are exact given the saved scores.
- **Some scripts carry the laptop's folder names:** `overnight/*.sh` and the `france_fix/` analyses, which ran once,
  interactively. The helper that copied files between machines is not included because it holds login details.
- **The v10a US/India blend is not yet a script in `src/`.** It is described in `submissions/v10a/finalize.json`
  (0.6 × ours + 0.4 × the second pipeline, then the list-mover overrides), and it moves into `src/` for the Round 2
  package. The v10b France addition is `src/france_recall.py`.

---

## Where the depth is

| Read | For |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every experiment with its held-out score, error breakdowns, the blocking audit, and the lessons section |
| [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) | Reproducing both submission files, step by step, with machines and run times |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Problem analysis, blocking, features, models, error analysis |
| [`submissions/LOG.md`](submissions/LOG.md) | All versions and their leaderboard results |
| `code/business_entity_resolution/france_fix/` | How each France rule was found and checked |
