# Business entity resolution: reproducing `matching_results.tsv` and `candidate_pairs.tsv`

This folder holds the full pipeline behind the final submission of team Mommy's Good Boys to the Amazon ML Challenge
2026: version `v10d`, public leaderboard 0.990565 (`submissions/LOG.md`). The commands below are listed in the order
they must run, from the raw dataset to the two output files.

- `matching_results.tsv`: one row per test Source 1 entity with its matched Source 2 / Source 3 records.
- `candidate_pairs.tsv`: every (Source 1 entity, record) pair that the final matching models score (definition in
  step 28).
- There are two ways to run it:
  - **From the raw data** (sections 5 and 6, steps 1-28). The logged laptop steps add up to about 29 h, including
    restarts, plus GPU runs on two other machines (section 1).
  - **From the saved intermediate files** (section 7, `python build_final.py`). This rebuilds v10a, v10b, v10c and v10d
    and compares their md5 with the uploaded files. On 27 Sep 2026 all five files were byte-identical, and the official
    validator printed PASS. The run took 304 s on the laptop (`src/logs/build_final.log`).
- "Exact" has a limited meaning here. Retraining a transformer on a GPU gives close, not identical, scores (section 8).
  Every step that starts from saved score files is exact.

In the submission zip this folder sits at `code/business_entity_resolution/`, next to `output/` (the two final files)
and `Documentation_template.md` (the methodology document, `docs/METHODOLOGY.md` in the repository).
- `EXPERIMENTS.md`, `submissions/LOG.md`, `submissions/<version>/finalize.json` and `docs/runbooks/`, cited below, are
  in the repository at those paths. In the zip, copies are in `docs/` of this folder (`docs/EXPERIMENTS.md`,
  `docs/LOG.md`, `docs/finalize/<version>.json`, `docs/runbooks/`).
- `src/logs/` holds the run logs cited below: the two stacker fits of step 27 and a run of `build_final.py`.

## 0. Terms

| Term | Meaning |
|---|---|
| S1 row | one Source 1 entity: a row of `{train,test}_source1.tsv` |
| record | one Source 2 or Source 3 row. The pipeline searches the S1 rows of the record's country for it |
| pair | one (record, S1 row) combination; in the code `q` is the record id and `s` the S1 id, both as integers |
| p1, p2 | probability that a pair is a true match, from stage 1 and from stage 2 (later stages overwrite `p2`) |
| close call | a record whose best p2 is in [0.01, 0.995], or whose 2nd candidate has p1 >= 0.2. The cross-encoders rescore it |
| cross-encoder | a transformer that reads the record's and the S1 row's "name \| address" text together and scores the pair |
| family | one cross-encoder model type trained as 3 fold models (e5-small, bge-reranker-v2-m3, e5-large) |
| fold model k | a model trained on the labelled close calls of the other two folds; it scores fold k out-of-fold and all test close calls |
| held-out half | the train S1 rows with `zlib.crc32(s1_id) % 1000 < 500`. All held-out scores are macro F0.5 on this half, searched against all train S1 rows of the country |
| second pipeline | this same code run with the wider `BER_V8=1` blocking on a lab GPU server (step 17) |
| expected-F0.5 rule | per S1 row, the set of records (in probability order) that maximises the expected F0.5; `pipeline.decide_expf` |
| pair set | a parquet table of (`s`, `q`) pairs that a step adds to or removes from a file; the final ones are in `sets/` |

## 1. Where each part ran

| Machine | Hardware | Steps |
|---|---|---|
| Team laptop (Kanav) | Intel i5-12450H, 16 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11, Python 3.12.10 | every step not listed below |
| Bhavya's workstation | NVIDIA RTX A6000 48 GB, Ubuntu | step 12 (bge-reranker-v2-m3, 3 fold models), step 14b (bge scores of the wide search's new pairs), step 21 (e5-large folds 0-1 and their wide-search scores) |
| Lab GPU server (shared; run by Gathik) | NVIDIA RTX PRO 6000 Blackwell 96 GB, Rocky Linux 8 | step 17 (the second pipeline), step 21 (e5-large fold 2 and its wide-search scores) |
| Harsh's laptop | NVIDIA RTX 4060 8 GB | a multilingual-e5-base 3-fold run used in earlier versions (v7g); **not** in the final file |

Files moved between machines by SSH/SFTP (`scp` works the same). The helper script used for this is not included
because it holds login details.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # pins the laptop's versions, incl. torch 2.7.1+cu118
```

- The laptop GPU driver supports CUDA 11.x only, hence `torch==2.7.1+cu118` and `xgboost==2.0.3`. On a newer driver,
  a CUDA 12 build of torch and a newer xgboost work with the same code. The other machines' versions are listed at
  the end of `requirements.txt`.
- A CUDA GPU is required: the XGBoost stages run with `device="cuda"`, and the transformers train and score on the
  GPU.
- Disk, measured on 27 Sep 2026 (hard links counted once):
  - the laptop's work folder was 98.5 GB, including experiment files;
  - pair features: train 19.2 GB, test 18.6 GB, wide US/India test search 25.8 GB (about 60 bytes per pair);
  - transformer models and scores 11.7 GB; embeddings 2.7 GB;
  - section 7 alone needs 2.8 GB of saved inputs and writes 1.5 GB.
- RAM: every step fits in 16 GB when steps run one at a time. Peak RAM of the long steps was not recorded. In
  section 7 the largest process was `stack/build_test.py` at 5.46 GB (`src/logs/build_final_memory.txt`).

## 3. Folders and environment variables

All commands run from `src/`. These variables set the folders:

```bash
cd code/business_entity_resolution/src
export BER_DATA=/path/to/student_resource/dataset   # holds train/ and test/ with the six *_source{1,2,3}.tsv + train_ground_truth.tsv
export BER_WORK=/path/to/work                        # every intermediate file
export BER_OUT=/path/to/output                       # the two final TSV files
export PYTHONIOENCODING=utf8
W=$BER_WORK
```

| Variable | Default | Read by | What it sets |
|---|---|---|---|
| `BER_DATA` | `<root>/student_resource/dataset` | `common.py` | the raw data |
| `BER_WORK` | `<root>/work` | `common.py` | intermediate files |
| `BER_OUT` | `<root>/output` | `common.py` | the two final files |
| `BER_ROOT` | three folders above `src/` | `common.py` | `<root>` above; `movers/build_movers.py` reads `<root>/submissions/` |
| `BER_SETS` | `../sets` | `build_final.py` | the pair sets and the stacker model file |
| `BER_BUILD` | `$BER_WORK/build_final` | `build_final.py` | its own intermediate files (1.5 GB) |
| `BER_VALIDATOR` | `<root>/student_resource/utils/validate_submission.py` | `build_final.py` | the official validator; skipped when absent |
| `BER_STACK_DIR` | `$BER_WORK/stack` (created when missing) | `stack/` | the stacker's own files |
| `BER_BLEND` | `$BER_WORK/blend9` | `stack/` | the output folder of `blend_second.py` |
| `BER_MOVERS` | `$BER_WORK/movers` | `stack/` | `tier12_removed.parquet`, `tier12_restored.parquet` |
| `BER_STACK_MODELS` | `$BER_STACK_DIR/lgb_models<tag>.pkl` | `stack/apply_test.py` | the stacker model file |
| `BER_SCRATCH` | `C:/ber_scratch` | `france_fix/`, `experiments/measure/` | scratch folder of the analysis and measurement scripts |
| `BER_V8` | unset | `normalize.py` (via `prep.py`), `candidates.py` | `1` = the second pipeline's cleaning and blocking (step 17 only) |
| `BER_BACKEND`, `BER_BACKEND2` | `xgb`, same as `BER_BACKEND` | `pipeline.py` | model type of stage 1 and stage 2 (`lgb` or `xgb`) |
| `BER_CE_*`, `BER_S3_*` | see the steps | `rerank.py` | cross-encoder folder, side, model, batch size; stage-3 families, tag, test score file |
| `BER_CALIB`, `BER_FR_LEGAL_VETO` | `calib.json`, unset | `finalize.py` | decision-rule file; `1` = France legal-form veto |
| `BER_CHUNK`, `BER_THREADS` | 250000, 11 | `candidates.py` | records per search chunk; threads |
| `BER_FREE_GB` | 64 | `topk5.py` (not on Windows) | free RAM in GB assumed when the Windows API is not available |

`<root>` is the folder that holds `code/`: the repository root, or the top of the submission zip.

- On Windows (Git Bash) use drive paths such as `E:/ber/work`; Python does not read `/e/...` paths.
- Use folder paths without spaces. The commands below leave `$W`, `$B` and `$BER_OUT` unquoted, so a path with a
  space is split into several arguments.
- Every long step skips work already on disk (chunk files, checkpoints, saved score chunks). A stopped step resumes
  when the same command is run again.

## 4. Pipeline in one picture

Step numbers in parentheses.

```
raw TSV -> cleaning (1) -> e5 embedding for non-Latin names (2)
        -> blocking + 55 pair features: train (3), test (5), wide US/India test search (13)
        -> stage 1 LightGBM (4): scores every blocked pair, keeps each record's top candidates
        -> stage 2 XGBoost with group features (4) -> p2 per pair
        -> close calls + stage-1 ranks 3-5 (6, 7)
        -> cross-encoders, 3 fold models per family: e5-small (11), bge-reranker-v2-m3 (12, 14), e5-large (21)
        -> stage 3 XGBoost per family (15, 21)
US/India:  blend 0.3 e5-small + 0.7 bge (15) -> 0.6 ours + 0.4 second pipeline (24)
           -> LightGBM stacker over both pipelines (27); list-mover overrides (23) after the blend and after the stacker
France:    e5-small / e5-base half models (8-10) + France rules and pair sets (16, 22, 25, 26, 28)
second pipeline: the same code with BER_V8=1 blocking on the lab server (17)
final files (28): matching_results.tsv, candidate_pairs.tsv
```

| Version | Public LB | Built by | Change |
|---|---|---|---|
| v9y | 0.989563 | steps 1-20 | wide US/India search, e5-small + bge folds, France rules, 16,421 second-pipeline pairs |
| v9z | not uploaded | step 22 | France: +5,883 missed matches, -73 same-stem descriptor swaps |
| v9zm | 0.989968 | step 23 | US/India list-mover fix (-839 / +832 pairs) on v9z |
| v10a | 0.990166 | step 24 | US/India probability = 0.6 ours + 0.4 second pipeline, then the list-mover overrides |
| v10b | 0.990475 | step 25 | +4,825 France matches from the second pipeline |
| v10c | not uploaded | step 26 | +802 France pairs from three pair sets |
| v10d | **0.990565** | steps 27-28 | US/India LightGBM stacker; -256 France pairs. Final upload |

Scores from `submissions/LOG.md`.

## 5. Commands, part A: the v9y file (steps 1-20)

These steps built v9y. Check on 27 Sep 2026: steps 16, 18 and 19, run from the saved intermediate files (step 17's
output as received), rebuilt both v9y files byte for byte. The md5 values were `2ebc84acd2cef360a9296ad6ee587b8c`
for `matching_results.tsv` (the same as `submissions/v9y/`) and `de40e20b1f9d975a4ee5307bf60324e3` for
`candidate_pairs.tsv`. The official `validate_submission.py --check-ids` printed PASS on them.

Run times below come from the logs in the laptop's work folder (`work/*.log`) and include restarts.

### Step 1. Cleaning (laptop CPU, minutes)

```bash
python learn_maps.py     # address short forms (rd -> road) and local-script state names, learned from TRAIN matches -> $W/maps.json
python prep.py           # clean all six source files -> $W/{train,test}_s{1,2,3}.parquet
```
Do not set `BER_V8` for these steps. It switches on the cleaning of step 17, which is a different run.

### Step 2. Embedding model for non-Latin names (laptop GPU; training ~15 min)

```bash
python embed.py train    # fine-tunes intfloat/multilingual-e5-small on (non-Latin record, S1) train pairs -> $W/e5_ft_addr/
python embed.py encode   # vectors for S1 rows of countries with non-Latin records + those records -> $W/emb/{train,test}_{s1,q}.npz
```
Training took 887 s (`embed_train2.log`); the encoding time was not logged.

### Step 3. Blocking and pair features for all of TRAIN (laptop, ~2.4 h)

```bash
python pipeline.py build train full          # -> $W/pairs/full/<country>_<chunk>.parquet (319.0M pairs, 30.9 per record)
```

### Step 4. Stage 1 and stage 2 (laptop; stage 1 CPU ~1.7 h, stage 2 GPU ~13 min)

```bash
BER_BACKEND=lgb python pipeline.py train full                        # stage 1 LightGBM -> $W/models/full/ (s1_f0..2, s1_full, OOF table)
BER_BACKEND=lgb BER_BACKEND2=xgb python pipeline.py train full cons  # stage 2 XGBoost with sibling + consensus features -> $W/models/full_cons/
```
- `BER_BACKEND=lgb` matters: the default of `pipeline.py` is XGBoost for stage 1, while the final models use LightGBM.
- The first command also trains a LightGBM stage 2 that later steps do not use. It must still run, because
  `topk5.py` (step 7) reads its `$W/models/full/result.json`.

### Step 5. Blocking for TEST (production settings) and test scores (laptop, ~2.2 h + ~1.3 h)

```bash
python pipeline.py build test test                                 # -> $W/pairs/test/ (307.2M pairs)
BER_OUT=$W/out_full_cons python pipeline.py predict full_cons test # -> $W/test_scores_full_cons.parquet (q, s, p1, p2)
```
- The build took 7,927 s over two runs: 3,377 s searching, 4,158 s computing features, the rest building the search
  indexes and loading data (`test_build.log`).
- `predict` also writes an intermediate submission into `$W/out_full_cons/`. It is not used. Its
  `candidate_pairs.tsv` holds every searched pair (307.2M, the raw blocking output before the stage-1 filter). It is
  not the submitted candidate file.

### Step 6. Close calls (laptop, seconds)

```bash
python rerank.py select full_cons    # -> $W/ce/{train,test}_rows.parquet, $W/ce/select.json
```
The top 2 candidates of each close call (section 0) are rescored by the transformers.

### Step 7. Stage-1 candidates ranked 3-5 of the close-call records (laptop, ~45 min)

```bash
python topk5.py train full    # -> $W/models/full/stage1_rank3_5.parquet (out-of-fold p1)
python topk5.py test full     # -> $W/test_rank3_5_test_full.parquet
BER_CE_DIR=$W/ce_x python rerank.py extras full_cons $W/ce    # close calls + ranks 3-5 -> $W/ce_x/{train,test}_rows.parquet
```
`topk5.py` reads free RAM through a Windows API call (`free_gb`) and waits while less than 4 GB is free. On Linux it reads
`BER_FREE_GB` instead (default 64).

### Steps 8-10. Transformers used for the France rows (laptop GPU, 8.2 h logged)

These three cross-encoders were trained on the two halves of train S1:
- side `a`: the half not used for evaluation;
- side `b`: the evaluation half;
- side `a2`: a larger model on side a's rows.

Their blend decides the France rows. France has no labels; step 16 explains why France keeps this older setting.

```bash
# step 8: e5-small, side a and side b
python rerank.py train intfloat/multilingual-e5-small              # -> $W/ce/model/
python rerank.py score                                             # -> $W/ce/{train,test}_ce.parquet
BER_CE_SIDE=b python rerank.py train intfloat/multilingual-e5-small
BER_CE_SIDE=b python rerank.py score                               # -> $W/ce/{train,test}_ce_b.parquet
python rerank.py stage3                                            # -> $W/ce/test_scores_ce_ab.parquet, oof_s3_ab, rule_ab.json
# step 9: e5-base, side a2, 420k training rows (16-bit frozen word table and batch 16 to fit 4 GB)
BER_CE_SIDE=a2 BER_CE_BS=16 BER_CE_HALF_EMB=1 BER_CE_LIMIT=420000 python rerank.py train intfloat/multilingual-e5-base
BER_CE_SIDE=a2 python rerank.py score
BER_S3_SIDES=a2 python rerank.py stage3                            # -> $W/ce/test_scores_ce_a2.parquet
# step 10: blend (weights chosen on held-out) and France min rule
python blend.py ab_a2 full_cons:ce:_ab full_cons:ce:_a2            # -> $W/test_scores_blend_ab_a2.parquet, $W/ce/rule_blend_ab_a2.json
python fr_minrule.py $W/test_scores_blend_ab_a2.parquet $W/test_scores_full_cons.parquet $W/ce/test_scores_blend_ab_a2_frmin.parquet
```
The 8.2 h (29,421 s) is the sum of the training, scoring, stage-3 and blend logs, including restarts. Part of side a
ran under Windows power throttling.

### Step 11. e5-small, 3 fold models (laptop GPU, ~6.2 h)

```bash
python ce_folds.py --model intfloat/multilingual-e5-small --name small
# -> models $W/ce/model_smallf{0,1,2}/, scores $W/ce_x/{train,test}_ce_smallf{0,1,2}.parquet
```
- Fold model k trains on every labelled close call whose record is in another pipeline fold (1.68M pairs). It then
  scores fold k (out-of-fold) and all test close calls.
- On a 4 GB GPU the word-embedding table is frozen and gradient checkpointing is on. Other settings: batch 32,
  learning rate 3e-5, 1 epoch, 128 tokens.
- The 6.2 h (22,252 s, `ce_folds_small.log`) includes one failed try on folds 0 and 2.

### Step 12. bge-reranker-v2-m3, 3 fold models (Bhavya's RTX A6000 48 GB, ~10-12 h)

On the laptop, pack the inputs: names and addresses of close-call records only, 329 MB, no raw files.
```bash
python bundle.py      # -> $W/bundle_ce_x.zip
```
On the GPU machine, unzip `bundle_ce_x.zip` in the root of a copy of this repository (the folder that holds `code/`),
so that `work/` is created there:
```bash
cd code/business_entity_resolution/src
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --smoke --no-freeze --batch 32 --lr 2e-5 --score-batch 512   # 3-min check
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
```
- Settings: full fine-tuning (word table trained), batch 64, learning rate 2e-5, 1 epoch (26,000 steps per fold).
- Time: about 3.4 h per fold for training plus scoring. Fold 2 ran 05:56 to 09:18 on 27 Sep (`work/f2_watch.log`).
  The card was shared with another user's job for part of the run.
- Copy back into `$W/ce_x/`: `train_ce_bgef{0,1,2}.parquet`, `test_ce_bgef{0,1,2}.parquet`.
- Keep the three models (`work/ce/model_bgef{0,1,2}/`) on the GPU machine for step 14b.

### Step 13. Wide test blocking for US and India (laptop, ~3.6 h + ~1.5 h)

```bash
python blocking_b2.py build      # combined name+address search max_df 20000, top 40 (production: 5000, top 20)
                                 # -> $W/pairs/test_b2/ (469.8M pairs; France chunks hard-linked from pairs/test)
python blocking_b2.py predict    # stage 1 + ranks 3-5 + stage 2 of full_cons -> $W/test_scores_full_cons_test_b2.parquet,
                                 #   $W/test_rank3_5_test_b2.parquet (prints a check: France p2 equals step 5 exactly)
python blocking_b2.py rows       # -> $W/ce_b2/test_rows.parquet (3,788,098 rows); train rows/scores linked from ce_x
```
`max_df` is the document-frequency cap: a term found in more than this many S1 rows is ignored during the search.

### Step 14. Transformer scores for the new pairs of the wide search

Pairs already scored in `ce_x` are copied; only the 585,295 new pairs are scored.

14a, laptop (e5-small, ~5 min per fold):
```bash
for k in 0 1 2; do
  BER_CE_DIR=$W/ce_b2 BER_CE_SIDE=f$k BER_CE_NAME=small BER_CE_REUSE=$W/ce_x python rerank.py score
done
```
14b, Bhavya's workstation (bge, ~5 min per fold). It needs two inputs:
- the **full** cleaned test files `work/test_s{1,2,3}.parquet` from step 1. The bundle of step 12 holds cut-down
  copies, and scoring with those gives empty texts;
- `work/ce_b2/test_rows.parquet` from step 13.

```bash
mkdir -p ../../../work/ce_b2          # put test_rows.parquet from the laptop's $W/ce_b2/ here
for k in 0 1 2; do
  ln -sf $PWD/../../../work/ce_x/train_ce_bgef$k.parquet ../../../work/ce_b2/train_ce_bgef$k.parquet
  BER_CE_DIR=../../../work/ce_b2 BER_CE_REUSE=../../../work/ce_x BER_CE_NAME=bge BER_CE_SIDE=f$k \
    BER_CE_MODEL=../../../work/ce/model_bgef$k BER_CE_SCORE_BS=1024 python rerank.py score
done
```
Copy back `work/ce_b2/test_ce_bgef{0,1,2}.parquet` into `$W/ce_b2/`.

### Step 15. Stage 3, blend and US/India decision (laptop, ~15 min)

```bash
for F in small bge; do
  BER_CE_DIR=$W/ce_b2 BER_S3_EXTRA=1 BER_S3_SIDES=${F}f0,${F}f1,${F}f2 BER_S3_TAG=_${F}folds \
    BER_S3_TEST=$W/test_scores_full_cons_test_b2.parquet python rerank.py stage3
done                                   # -> $W/ce_b2/test_scores_ce_{small,bge}folds.parquet (+ held-out scores in the log)
python blend.py v7p full_cons:ce_b2:_smallfolds full_cons:ce_b2:_bgefolds
                                       # weights 0.3 / 0.7, the best of 0.3/0.7, 0.5/0.5, 0.7/0.3 on held-out (bge alone
                                       # scored the same) -> $W/test_scores_blend_v7p.parquet, $W/ce/rule_blend_v7p.json
python fr_minrule.py $W/test_scores_blend_v7p.parquet $W/test_scores_full_cons.parquet $W/ce/test_scores_blend_v7p_frmin.parquet
BER_CALIB=$W/ce/rule_blend_v7p.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $W/out_v7p_all $W/ce/test_scores_blend_v7p_frmin.parquet
```
Only the US and India rows of `$W/out_v7p_all/matching_results.tsv` are used. They equal the US/India rows of
`submissions/v7p/` (checked on 27 Sep 2026).

### Step 16. France rows (laptop, ~1 min)

The France rows start from the scores of step 10 and apply three pair sets from `../sets/`, in this order:

| Set | Pairs | Change | Defined by (analysis scripts in `src/france_fix/`) |
|---|---|---|---|
| `desc_veto_set.parquet` | 22,436 | p2 = 0 | `namechg/a1_build.py` ... `a11_veto.py`: accepted France pairs whose record adds or swaps in a descriptor word (amicale, comite, ecole, club, centre, ...), synonym and stem swaps excluded |
| `noise_add_set.parquet` | 9,800 | p2 = max(p2, 0.95) | `build/b4_addset.py`, `build/b8_mkscores_v2.py`: rejected same-address pairs whose added or swapped word is a noise word, or groupe/developpement/france where the v7m probe had restored the pair |
| `fp_veto_set.parquet` | 1,145 | p2 = 0 | `artifacts/fp/fp1_dec.py` ... `fp10_veto.py`: accepted pairs in three groups (column `tier`). A, 846 pairs: legal form added and house number moved up by 1-20 (the look-alike signature). B, 232 pairs: word added, swapped or mistyped and house number moved up by 1-20. C, 67 pairs: all-lowercase record with a name change |

The parquet files are not in git; they ship in the submission zip (`sets/README.md`). Apply them and decide the
France rows with the rule of step 10:
```bash
FS=../sets SC=$W/ce/test_scores_blend_ab_a2_frmin.parquet DST=$W/test_scores_france_final.parquet python - <<'EOF'
import os, polars as pl
fs, b = os.environ["FS"], pl.read_parquet(os.environ["SC"])
def mark(b, f):
    return b.join(pl.read_parquet(f, columns=["q", "s"]).with_columns(hit=pl.lit(True)), on=["q", "s"], how="left").with_columns(pl.col("hit").fill_null(False))
b = mark(b, f"{fs}/desc_veto_set.parquet").with_columns(p2=pl.when("hit").then(0.0).otherwise("p2").cast(pl.Float32)).drop("hit")
b = mark(b, f"{fs}/noise_add_set.parquet").with_columns(p2=pl.when("hit").then(pl.max_horizontal("p2", pl.lit(0.95))).otherwise("p2").cast(pl.Float32)).drop("hit")
b = mark(b, f"{fs}/fp_veto_set.parquet").with_columns(p2=pl.when("hit").then(0.0).otherwise("p2").cast(pl.Float32)).drop("hit")
b.select("q", "s", "p1", "p2").write_parquet(os.environ["DST"])
EOF
BER_CALIB=$W/ce/rule_blend_ab_a2.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $W/out_france $W/test_scores_france_final.parquet
```
- Only the France rows of `$W/out_france/matching_results.tsv` are used.
- `BER_FR_LEGAL_VETO=1` also sets p2 = 0 for France pairs whose two raw names both carry a legal form (SARL, SAS,
  SA, EURL, ...) with none in common.

To re-derive the three sets instead of using the shipped files, run the scripts named in the table in file-name
order.
- They were run once, interactively, on the laptop.
- Each has its input and output folders as constants at the top of the file (`FR`, `OUT`, `NC`, `RD`, `F`, `T`, `W`),
  under the scratch folder `$BER_SCRATCH` (default `C:/ber_scratch`) or `$BER_WORK`.
- Their inputs are best-candidate tables built by `france_fix/tables/fr1.py`, `fr4.py` and `fr8.py`:
  `fr_top.parquet` for France test records, `usi_top.parquet` for labelled US/India train records, and
  `pairs_<version>.parquet` for the France pairs of earlier submissions.
- Those tables are built from the step 5 and step 10 test scores, the out-of-fold train scores
  (`$W/models/full_cons/oof.parquet`, `$W/ce_x/oof_s3_smallfolds.parquet`) and earlier submission files (v7ens and the
  France probe versions listed in the methodology document).

### Step 17. Second pipeline: candidate generation and matching (lab GPU server, run by Gathik; `src/runners/run_v9.sh`)

A second full run of this pipeline with the audited blocking, an XGBoost stage 1 and 3 bge-reranker-v2-m3 fold
models. It needs its own work folder because its cleaning differs from step 1.
- `BER_V8=1` blocking: name search max_df 10000, no-address name search top 100, word searches with min_df 1, and a
  dense e5 top-10 search for every record.
- `BER_V8=1` cleaning: India state codes and spelled-out ordinals.

```bash
export BER_V8=1 BER_WORK=~/ber/w BER_DATA=~/ber/dataset BER_OUT=~/ber/out/v9 BER_CE_DIR=~/ber/w/ce_x
export CUDA_VISIBLE_DEVICES=2 BER_THREADS=24 POLARS_MAX_THREADS=24 BER_CHUNK=125000   # shared server limits
# copy maps.json and e5_ft_addr/ from the laptop's $W into ~/ber/w first
python prep.py                                  # BER_V8=1 cleaning (state codes, ordinals)
python embed.py encode_all                      # e5 vectors of every S1 and S2/S3 row (~40 min)
python pipeline.py build train full             # 417.5M train pairs, 40.5 per record
python pipeline.py build test test
python pipeline.py train full cons              # BER_BACKEND unset: XGBoost stage 1 -> $BER_WORK/models/full_xgb_cons/
python pipeline.py predict full_xgb_cons test
python rerank.py select full_xgb_cons           # close calls -> $BER_CE_DIR
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --dir $BER_CE_DIR --no-freeze --batch 64 --lr 2e-5 \
    --score-batch 1024 --train-rows 1000000
BER_S3_SIDES=bgef0,bgef1,bgef2 python rerank.py stage3
cp $BER_CE_DIR/rule_bgef0bgef1bgef2.json $BER_WORK/models/full_xgb_cons/rule_v9.json
python fr_minrule.py $BER_CE_DIR/test_scores_ce_bgef0bgef1bgef2.parquet $BER_WORK/test_scores_full_xgb_cons.parquet \
    $BER_CE_DIR/test_scores_v9_frmin.parquet
BER_CALIB=rule_v9.json BER_FR_LEGAL_VETO=1 python finalize.py full_xgb_cons $BER_OUT $BER_CE_DIR/test_scores_v9_frmin.parquet
```
`src/runners/run_v9.sh` is the scripted version, with a smoke test, retries and waits for free RAM on the shared
server.

Copy these files to the laptop; the later steps read them under these names:

| Server file | Laptop file | Used by |
|---|---|---|
| `$BER_OUT/matching_results.tsv` | `$W/gathik/v8_matching_results.tsv` (historical name) | steps 18, 19, 25, 28 |
| `$BER_CE_DIR/test_scores_ce_bgef0bgef1bgef2.parquet` | `$W/gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet` | steps 24, 27 |
| `$BER_CE_DIR/test_scores_v9_frmin.parquet` | `$W/gathik/v9/ce_x_test_scores_v9_frmin.parquet` | step 25 |
| `$BER_WORK/test_scores_full_xgb_cons.parquet` | `$W/gathik/v9/test_scores_full_xgb_cons.parquet` | step 27 |
| `$BER_CE_DIR/oof_s3_bgef0bgef1bgef2.parquet` | `$W/gathik/v9/ce_x_oof_s3_bgef0bgef1bgef2.parquet` | step 27 (held-out) |
| `$BER_WORK/models/full_xgb_cons/oof.parquet` | `$W/gathik/v9/models_full_xgb_cons_oof.parquet` | step 27 (held-out) |

- Matched pairs of this run: US 2,255,299, India 2,748,510, France 853,628 (`EXPERIMENTS.md`).
- Held-out macro F0.5: 0.99183 with all close calls rescored.
- The transformer stage ran 27 Sep 10:27-12:54 IST. The full server run time was not recorded.
- `src/experiments/v8.py` is an earlier streamed build on the same server (files v8a/v8b). It is not part of the
  final file.

### Step 18. v9y matching file (laptop, seconds)

```bash
python assemble_final.py $W/out_v7p_all/matching_results.tsv $W/out_france/matching_results.tsv $BER_OUT \
    $W/gathik/v8_matching_results.tsv $W/test_scores_blend_v7p.parquet
```
- US/India rows come from step 15 and France rows from step 16.
- Added: the step 17 pairs that are absent from our candidate lists, for records our file leaves unmatched (16,421
  pairs: India 12,939, US 3,482).
- Output: `$BER_OUT/matching_results.tsv` (5,846,292 matched pairs; 1,632,486 of 1,732,544 S1 rows non-empty).

### Step 19. v9y candidate file (laptop, seconds)

```bash
python make_candidates.py $W/test_scores_blend_v7p.parquet $BER_OUT/matching_results.tsv $BER_OUT \
    $W/gathik/v8_matching_results.tsv
```
- The file holds every (record, S1) pair of the final stage-2/stage-3 score table: each record's top stage-1
  candidates (the input of stage 2), the stage-1 ranks 3-5 of close-call records, and the 16,421 step 17 pairs of
  step 18.
- The script stops with an error if any matched pair is not a candidate.
- v9y result: 11,658,294 pairs, 6.73 per S1 row (US 6.29, India 6.70, France 7.94), 377 S1 rows without candidates.
  The final file of step 28 is larger (12,249,116 pairs), because it also holds the pairs only the second pipeline
  scored.

### Step 20. Checks, and keeping the v9y file

```bash
python check_submission.py $BER_OUT/matching_results.tsv             # fast check of every leaderboard rule
python <student_resource>/utils/validate_submission.py -m $BER_OUT/matching_results.tsv \
    -c $BER_OUT/candidate_pairs.tsv -t $BER_DATA/test --check-ids     # official validator: PASS
mkdir -p $W/out_v9y && cp $BER_OUT/matching_results.tsv $BER_OUT/candidate_pairs.tsv $W/out_v9y/
```
Step 23 reads the v9y file, and step 28 overwrites `$BER_OUT`.

## 6. Commands, part B: from v9y to the final v10d (steps 21-28)

These steps follow the order in which the versions were built. `B` is the folder for their intermediate files, and
`S` is the pair-set folder:

```bash
B=$W/build_final; S=../sets; mkdir -p $B
```

Each set in `sets/` is listed in `sets/README.md` with its row count, md5, meaning and the scripts that chose it.

### Step 21. e5-large family (GPU machines; laptop stage 3, ~5 min)

A third cross-encoder family, intfloat/multilingual-e5-large. The fixed blends do not use it: held-out, all close
calls rescored, the three families scored 0.98946 against 0.98950 for v7p (`EXPERIMENTS.md`). The v10d stacker
(step 27) uses its stage-3 probability as feature `c`.

On the GPU machines, with the inputs of step 12 and the full test files of step 14b:
```bash
# Bhavya's workstation: folds 0 and 1; lab server: --folds 2
python ce_folds.py --model intfloat/multilingual-e5-large --name e5l --folds 0,1 --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
# -> work/ce_x/{train,test}_ce_e5lf<k>.parquet, models work/ce/model_e5lf<k>/
# scores for the new pairs of the wide search, as step 14b (k = the machine's folds):
for k in 0 1; do
  ln -sf $PWD/../../../work/ce_x/train_ce_e5lf$k.parquet ../../../work/ce_b2/train_ce_e5lf$k.parquet
  BER_CE_DIR=../../../work/ce_b2 BER_CE_REUSE=../../../work/ce_x BER_CE_NAME=e5l BER_CE_SIDE=f$k \
    BER_CE_MODEL=../../../work/ce/model_e5lf$k BER_CE_SCORE_BS=1024 python rerank.py score
done
```
On the laptop, after copying `ce_x/{train,test}_ce_e5lf{0,1,2}.parquet` into `$W/ce_x/` and
`ce_b2/test_ce_e5lf{0,1,2}.parquet` into `$W/ce_b2/`:
```bash
for k in 0 1 2; do cp $W/ce_x/train_ce_e5lf$k.parquet $W/ce_b2/; done
BER_CE_DIR=$W/ce_b2 BER_S3_EXTRA=1 BER_S3_SIDES=e5lf0,e5lf1,e5lf2 BER_S3_TAG=_e5lfolds \
  BER_S3_TEST=$W/test_scores_full_cons_test_b2.parquet python rerank.py stage3
# -> $W/ce_b2/test_scores_ce_e5lfolds.parquet (test) and $W/ce_b2/oof_s3_e5lfolds.parquet (held-out)
```
- Settings are those of step 12: full fine-tuning, batch 64, learning rate 2e-5, 1 epoch.
- Held-out AUC of each fold model on its out-of-fold rows: 0.98926 / 0.98952 / 0.98922 (bge 0.99022 / 0.99034 /
  0.98995; `EXPERIMENTS.md`).
- The GPU run times were not recorded. The scripted versions are `src/runners/e5l_after.sh` and
  `src/runners/b2_e5l_remote.sh` (GPU machine) and `src/runners/e5l_final.sh` (laptop).

### Step 22. v9z France rows (laptop, ~20 s)

v9z adds 5,883 missed France matches (`recall_add_set`, p2 = 0.95) and removes 73 France matches whose record swaps a
descriptor word for another word with the same 5-letter stem (`moreveto_stem2_set`, p2 = 0).

```bash
BER_BUILD=$B python build_final.py step v9z_scores      # step 16 scores + the two sets -> $B/test_scores_v9z_france.parquet
BER_CALIB=$W/ce/rule_blend_ab_a2.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $B/v9z_all $B/test_scores_v9z_france.parquet
```
- Only the France rows of `$B/v9z_all/matching_results.tsv` are used (852,676 France pairs). v9zm and v10a keep them
  unchanged.
- The 2 recovered pairs that are missing from the score table are appended with p1 = p2 = 0.95.
- The scripts that chose the two sets are in `src/france_fix/v9z/` (its `README.md` lists them).

### Step 23. US/India list movers (v9zm) (laptop, ~9 min for `v7q` from `combo_v7q.log`, 16 s for the movers)

The wide search (step 13) and the production search give different candidate lists. Where the two builds disagree
on a US/India pair, the list movers let a transformer-only probability decide, because it does not depend on the list.
- Removed: 839 pairs that only the wide build accepted.
- Restored: 832 pairs that only the production build accepted, for records v9y leaves unmatched.
- Rules: `movers/build_movers.py` docstring and `sets/README.md`.

`build_movers.py` compares the US/India rows of three earlier files: v7p (step 15), v7q (below) and v9y (step 18).
It reads them from `<root>/submissions/<name>/matching_results.tsv`. `v7q` is the same two families on the
production candidate lists:
```bash
for F in small bge; do
  BER_CE_DIR=$W/ce_x BER_S3_EXTRA=1 BER_S3_SIDES=${F}f0,${F}f1,${F}f2 BER_S3_TAG=_${F}folds python rerank.py stage3
done
python blend.py v7q full_cons:ce_x:_smallfolds full_cons:ce_x:_bgefolds    # -> $W/test_scores_blend_v7q.parquet, $W/ce/rule_blend_v7q.json
python fr_minrule.py $W/test_scores_blend_v7q.parquet $W/test_scores_full_cons.parquet $W/ce/test_scores_blend_v7q_frmin.parquet
BER_CALIB=$W/ce/rule_blend_v7q.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $W/out_v7q $W/ce/test_scores_blend_v7q_frmin.parquet
```
On the machine that built the uploads, `submissions/` holds the three files (git ignores `*.tsv`, so a clone does not
have them):
```bash
python movers/build_movers.py v7p v7q v9y $W/movers    # -> $W/movers/tier12_removed.parquet, tier12_restored.parquet
```
Elsewhere (for example from the zip), give it copies through `BER_ROOT`. Never copy into the repository's
`submissions/`: its v7p and v7q files hold other France rows.
```bash
M=$W/movers_in; for v in v7p v7q v9y; do mkdir -p $M/submissions/$v; done
cp $W/out_v7p_all/matching_results.tsv $M/submissions/v7p/
cp $W/out_v7q/matching_results.tsv $M/submissions/v7q/
cp $W/out_v9y/matching_results.tsv $M/submissions/v9y/
BER_ROOT=$M python movers/build_movers.py v7p v7q v9y $W/movers
```
- The US/India rows of `$W/out_v7p_all` and `$W/out_v7q` equal those of `submissions/v7p/` and `submissions/v7q/`
  (checked on 27 Sep 2026). `src/runners/build_combo.sh v7q small bge` is the scripted version of the `v7q` lines.
- Both commands above, run on 27 Sep 2026, wrote `tier12_*.parquet` files byte-identical to the shipped
  `sets/tier12_*.parquet`. Steps 24 and 27 can use either folder.

### Step 24. v10a: US/India blend with the second pipeline (laptop, ~25 s)

```bash
python blend_second.py $B/blend $W/movers          # or ../sets: the same two mover files
BER_BUILD=$B python build_final.py step v10a         # US/India matches of the blend + France rows of step 22 -> $B/v10a/matching_results.tsv
```
- For each US/India pair, p2 = 0.6 × ours (step 15) + 0.4 × the second pipeline's (step 17) where both scored the
  pair; otherwise the one that scored it.
- Then the list-mover overrides: removed pairs get p2 = 0; restored pairs get p2 = max(p2, 0.95).
- Then the expected-F0.5 rule with floor 0.5, alpha 1.0.
- Held-out: ours alone 0.98950, second pipeline alone 0.99183, blend 0.99205.
- `blend_second.py` also writes `cand_union.parquet` (12,243,553 pairs): every pair of `test_scores_blend_v7p.parquet`,
  every US/India pair the second pipeline scored, and the restored list-mover pairs. It is the candidate list of
  v10a-v10d.

### Step 25. v10b: France matches from the second pipeline (laptop, ~10 s)

```bash
python france_recall.py $B/v10a/matching_results.tsv $W/gathik/v8_matching_results.tsv \
    $W/gathik/v9/ce_x_test_scores_v9_frmin.parquet $B/v10b
```
- It adds the second pipeline's France matches that our France candidate lists never contained, for records v10a
  leaves unmatched: 4,825 of 7,266.
- Kept pairs have no word-level name change, no legal-form change or addition, no house number moved up, and a second
  pipeline probability of at least 0.8.
- The same kind of pair on labelled US/India data is 98.8% true (`EXPERIMENTS.md`).

### Step 26. v10c: three France pair sets (laptop, ~5 s)

```bash
python apply_pair_sets.py $B/v10b/matching_results.tsv $B/v10c \
    +$S/fr_typo_safe.parquet +$S/fr_same_address_safe.parquet +$S/fr_amp_safe.parquet
```
- Adds 802 France pairs: 416 garbled-word typos, 298 same-address acronym or noise-word names, and 88 names that write
  '&' as 'et' or '+'.
- A `+` set adds pairs, a `-` set removes them. On Git Bash, set `MSYS_NO_PATHCONV=1` when such an argument is a
  Windows absolute path.

### Step 27. v10d: US/India LightGBM stacker (laptop; each `fit.py` run ~3 min, test part ~2 min)

A LightGBM model rescores every US/India pair from 24 per-pair features of both pipelines. Order, features and
held-out results are in `stack/README.md`.

```bash
cd stack
export BER_STACK_DIR=$B/stack BER_BLEND=$B/blend BER_MOVERS=$W/movers     # without step 23: BER_MOVERS=../../sets
python build_ho.py                                             # held-out feature table -> $BER_STACK_DIR/ho.parquet, ev.parquet, truth_ev.parquet
DROP=nq,rank_s,margin_s TAG=_rob python fit.py gbdt            # 2 models, one per part of the held-out half
DROP=nq,rank_s,margin_s TAG=_robtr FITON=1 python fit.py gbdt  # 2 models fitted on the training half
python score_avg.py                                            # held-out check of the 4-model mean -> res_avg.json
python merge_models.py _rob _robtr _avg                        # -> $BER_STACK_DIR/lgb_models_avg.pkl
python build_test.py                                           # test feature table -> te.parquet
python apply_test.py _avg                                      # -> usi_pairs_avg.parquet (list movers after the model, rule floor 0.5)
cd ..
```
- Model settings: 15 leaves, 300 rounds, learning rate 0.05. The score is the mean of the 4 models
  (`submissions/v10d/finalize.json`).
- Held-out: +0.000209 over the v10a blend (0.99205 to 0.99226).
- Test: US +3,733 / -841 pairs, India +3,288 / -434 pairs against v10c.
- Each `fit.py` run took about 3 min: 180 s and 156 s, each model 20-42 s (`src/logs/stack_fit_rob.log`,
  `stack_fit_robtr.log`; the held-out results are next to them as `stack_res_*.json`). `build_ho.py` and
  `score_avg.py` were not timed. In the run of section 7, `build_test.py` took 32 s and `apply_test.py` 92 s
  (`src/logs/build_final.log`).
- The two `fit.py` commands were reconstructed from the saved models' feature lists and file tags; they were not
  logged. `sets/lgb_models_avg.pkl` is the model file of the uploaded v10d. To use it, set
  `BER_STACK_MODELS=../../sets/lgb_models_avg.pkl` and run only `build_test.py` and `apply_test.py _avg`.

### Step 28. v10d: France veto, final files and checks (laptop, ~15 s + ~2.5 min for the official validator)

```bash
BER_BUILD=$B python build_final.py step v10d
python check_submission.py $BER_OUT/matching_results.tsv
python <student_resource>/utils/validate_submission.py -m $BER_OUT/matching_results.tsv \
    -c $BER_OUT/candidate_pairs.tsv -t $BER_DATA/test --check-ids
```
`build_final.py step v10d` does four things:
- It checks that the stacker's difference from the v10c US/India pairs equals `sets/usi_stack_add.parquet` (7,021
  pairs) and `sets/usi_stack_remove.parquet` (1,275 pairs).
- It removes the 256 pairs of `fr_street_veto.parquet` from the v10c France rows. These pairs have the same generic
  name and house number but a completely different street, and both bge families score them below 0.3.
- It writes `$BER_OUT/matching_results.tsv`: the stacker's US/India rows and the France rows. The file has 5,862,410
  matched pairs (US 2,255,045, India 2,749,318, France 858,047).
- It writes `$BER_OUT/candidate_pairs.tsv` with `make_candidates.py`. Candidates are the pairs of `cand_union.parquet`
  (12,243,553), plus the pairs of the second pipeline's matching file that the final file also matches (5,340 new),
  plus the pairs of the same-address rule (`fr_same_address_safe`) that the final file matches (223 new). The file has
  12,249,116 pairs, 7.07 per S1 row, and 184 S1 rows without candidates. Every matched pair is a candidate.

The candidate file, measured on 27 Sep 2026 from `submissions/v10d/candidate_pairs.tsv` (`src/experiments/measure/test_candidates.py`):

| | US | India | France | All |
|---|---:|---:|---:|---:|
| candidate pairs | 4,345,438 | 5,837,449 | 2,066,229 | 12,249,116 |
| per S1 row | 6.55 | 7.21 | 7.96 | 7.07 |
| per record | 1.14 | 1.24 | 1.44 | 1.23 |

On labelled held-out data, the same candidate definition holds 98.94% of US/India true pairs. The production search
before the stage-1 cut holds 98.51%. Both were measured on 27 Sep 2026 from the saved out-of-fold score files of
steps 3, 4, 7 and 17 (`src/experiments/measure/heldout_recall.py`). The wide search of step 13 exists for test only, so its effect is not in the held-out figure.

## 7. One command from the saved files: `build_final.py`

```bash
python build_final.py      # BER_SETS (default ../sets), BER_BUILD (default $BER_WORK/build_final, 1.5 GB)
```

It runs steps 22 and 24-28 in order, each in its own process. The step 16 France scores are rebuilt inside step 22.
Then it runs `check_submission.py` and the official validator, and prints the md5 of the rebuilt files next to the
uploaded ones. It exits with 1 if any differs.
- **Not run:** the fitting part of step 27 (`build_ho.py`, `fit.py`, `score_avg.py`, `merge_models.py`). It uses
  `sets/lgb_models_avg.pkl` instead.
- **Not run:** step 23. It uses the shipped `sets/tier12_*.parquet`.

Inputs from `$BER_WORK` (2.8 GB, 16 files):

| File | From step |
|---|---|
| `test_s1.parquet`, `test_s2.parquet`, `test_s3.parquet` | 1 |
| `test_scores_full_cons.parquet`, `ce_x/test_rows.parquet` | 5, 7 |
| `ce/test_scores_blend_ab_a2_frmin.parquet`, `ce/rule_blend_ab_a2.json` | 10 |
| `test_scores_full_cons_test_b2.parquet` | 13 |
| `ce_b2/test_scores_ce_smallfolds.parquet`, `ce_b2/test_scores_ce_bgefolds.parquet`, `test_scores_blend_v7p.parquet` | 15 |
| `gathik/v8_matching_results.tsv`, `gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet`, `gathik/v9/ce_x_test_scores_v9_frmin.parquet`, `gathik/v9/test_scores_full_xgb_cons.parquet` | 17 |
| `ce_b2/test_scores_ce_e5lfolds.parquet` | 21 |

Plus `$BER_DATA/test/test_source{1,2,3}.tsv`, which `check_submission.py` and the official validator read.

Check on 27 Sep 2026 (laptop, 304 s, empty build folder, largest process 5.46 GB; `src/logs/build_final.log`,
`src/logs/build_final_memory.txt`):

| File | Rebuilt md5 = uploaded md5 |
|---|---|
| v10a `matching_results.tsv` | `c55eb79152a49674df6154c00edcc672` |
| v10b `matching_results.tsv` | `76cc451a73f6adb3cf81c9c8f7eab8ba` |
| v10c `matching_results.tsv` | `6fc63685e4ef18a00923f165675d0bad` |
| v10d `matching_results.tsv` | `18412329111b5d9c43df3a58df0574d1` |
| v10d `candidate_pairs.tsv` | `a2ddcb5d26ede94f780fd2c4d87519a7` |

- `check_submission.py` printed PASS, and the official validator (`--check-ids`) printed PASS.
- The step 16 France scores it rebuilds hold the same pairs and p1/p2 values as the saved
  `work/frfix2/test_scores_v9b_fpveto.parquet`; only the row order differs.
- The v9z score file equals the saved `work/frfix3/test_scores_v9y_combo.parquet`.

## 8. Limits of this reproduction

- **GPU training is not bit-for-bit repeatable.** Retrained transformers give slightly different scores, so a full
  rerun of steps 2, 8-12, 14, 17 and 21 gives files close to, not identical with, the uploaded ones. Everything that
  starts from the saved score files is exact (section 7).
- **Three machines.** Steps 12, 14b, 17 and 21 ran on machines other than the laptop (section 1). Step 17's output was
  received as files.
- **The stacker fit commands were reconstructed** from the saved models' feature lists, the file tags and the logs.
  They were not logged, so the model file ships as a saved output (`sets/lgb_models_avg.pkl`).
- **The France pair sets come from analysis scripts that were run once, interactively.** Their input and output
  folders are constants at the top of each file, under `$BER_SCRATCH`. They were not re-run for this package; the
  sets ship as saved outputs.
  - The last 2-pair drop in `fr_street_veto` was a command typed during the run. It removed the 2 pairs whose S1 row
    would have lost every match.
  - The two `usi_stack_*` sets were also typed commands. They are used only as a check.
- **Step 23 reads three earlier submission files.** They exist only in the team's local `submissions/` folder (git
  ignores `*.tsv`, and the zip has no `submissions/`); step 23 shows how to rebuild them from the outputs of steps 15,
  20 and 23. The two mover sets also ship in `sets/`.
- **Scripts in `src/runners/` contain the laptop's and the servers' folder names.** They are the shell scripts that ran
  steps 13-15, 17 and 21 unattended on 26-27 Sep, and some call the SSH helper that is not included.
- **`topk5.py` reads free RAM through a Windows API call** (step 7); on other systems it assumes `BER_FREE_GB` (default 64).
- **The zip holds neither the dataset nor the 2.8 GB of saved inputs of section 7.** A run from the zip alone starts
  at step 1.

## 9. Source files

Final pipeline (`src/`):

| File | Role |
|---|---|
| `common.py` | folders, TSV reading, integer ids, macro F0.5 |
| `normalize.py`, `learn_maps.py`, `prep.py` | cleaning; maps learned from train matches only |
| `embed.py` | fine-tuned multilingual-e5-small; vectors for the embedding search |
| `candidates.py` | blocking (4 searches) and the 55 pair features |
| `pipeline.py` | build / train / predict: stage 1, stage 2, grouped 3-fold validation, expected-F0.5 decision |
| `topk5.py` | stage-1 ranks 3-5 for close-call records |
| `rerank.py` | close calls, cross-encoder training and scoring, stage 3 |
| `ce_folds.py`, `bundle.py` | 3-fold cross-encoder run for a GPU machine; its input bundle |
| `blocking_b2.py` | wide combined search for US/India test records |
| `blend.py` | weighted blend of stage-3 families, decision rule chosen on held-out |
| `fr_minrule.py`, `finalize.py` | France: transformer may only lower p; legal-form veto; per-country decision |
| `assemble_final.py`, `make_candidates.py` | final matching file and candidate file |
| `check_submission.py` | fast rule check |
| `france_fix/artifacts/census/ops.py` | step 25: detects the change between a record and an S1 row (word swaps, legal forms, house number); imported by `france_recall.py` |
| `movers/build_movers.py` | step 23: US/India list movers (`tier12_removed`, `tier12_restored`) |
| `blend_second.py` | step 24: 0.6 × ours + 0.4 × the second pipeline, list-mover overrides, decision |
| `france_recall.py` | step 25: France matches from the second pipeline |
| `apply_pair_sets.py` | steps 26 and 28: adds and removes pair sets |
| `stack/` | step 27: US/India LightGBM stacker (`stack/README.md`) |
| `build_final.py` | section 7: rebuilds v10a-v10d from the saved files and checks the md5 |
| `logs/` | run logs: the two stacker fits of step 27 (`stack_fit_*.log`, `stack_res_*.json`) and a `build_final.py` run (`build_final.log`) |

Other folders:

| Folder | What it holds |
|---|---|
| `sets/` (next to `src/`) | the 13 pair sets and the stacker model file of the final build; `sets/README.md` lists rows, md5 and origin. The parquet and pkl files are in the zip, not in git |
| `src/france_fix/` | index in `src/france_fix/README.md`. France analyses: census of the changes the data generator makes (`artifacts/census/`), look-alike and missed-match hunts, leaderboard fits (`lbcal/`), the step 16 sets (`namechg/`, `build/`, `artifacts/fp/`), their input tables (`tables/`), the v9z sets (`v9z/`), the v10c/v10d sets (`final_hunt/`); `france_cal.py` and `fr_restore.py` (probe versions v7j and v7m) sit at its top |
| `src/runners/` | shell scripts: `run_v9.sh` (step 17, started by `v9_after_v8.sh`); `run_v8.sh`, `run_v8_resume.sh`, `v8_guard.sh` (the earlier `experiments/v8.py` build); `watchdog.sh` (stops a test build when free RAM runs low); `b2_*.sh`, `e5l_*.sh`, `f2_watch.sh`, `save_ce.sh` (unattended laptop and GPU-machine runs of 26-27 Sep); `build_combo.sh` (a submission from 3-fold families, used for v7f, v7g and v7q) |
| `src/experiments/` | experiments not needed for the final files. Each adds `src/` to its import path, so it runs from any folder. `measure/` holds the scripts behind the numbers the documentation measured on 27 Sep 2026 (`measure/README.md`) |

The files in `src/experiments/` are:
- `v8.py`, the earlier streamed build on the lab server, and `dl_matcher_exp.py`, the transformer as final matcher
  (`v8.py` imports its cross-encoder class);
- `avg_family.py`, `blocking_exp.py`, `blocking_variants.py`, `calib.py`, `cmp_fold0.py`,
  `combined_blocking_exp.py`, `decoy_sim.py`, `dense_blocking_exp.py`, `dlx_analyse.py`, `embed_all.py`,
  `error_analysis.py`, `error2.py`, `evaluate.py`, `fr_desc.py`, `fr_num.py`, `hybrid_b2.py`, `merge_country.py`,
  `name_cap_exp.py`, `name_search_diag.py`, `ordinal_exp.py`, `selftrain.py`, `splice_country.py`,
  `stage3_joint.py`, `testlike.py`, `testlike_build.py`.

## 10. Models and licences

| Model | Licence | Parameters | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M (counted from the saved weights) | embedding search (fine-tuned); cross-encoders A, B and 3 folds |
| intfloat/multilingual-e5-base | MIT | 278.0M (counted) | cross-encoder A2 (France rows) |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M (counted) | cross-encoder, 3 folds (US/India); cross-encoder of the second pipeline |
| intfloat/multilingual-e5-large | MIT | ~560M (published figure; the weights stayed on the GPU machines) | cross-encoder, 3 folds; its stage-3 probability is a stacker feature |
| LightGBM 4.7.0 / XGBoost 2.0.3 | MIT / Apache-2.0 | tree ensembles | stages 1-3, the v10d stacker |

- The largest model has under 0.6B parameters; the limit is 8B.
- No external data, APIs or lookup services are used.
- The learned tables and tree models are fitted on the provided training data only. The transformers are the public
  checkpoints above, fine-tuned on the provided training data.
