# Business entity resolution: reproducing `matching_results.tsv` and `candidate_pairs.tsv`

This folder holds the full pipeline that produced the final submission (version `v9y`, public leaderboard 0.989563,
27 Sep 2026). Commands below are listed in the order they must run, from the raw dataset to the two output files.

> **Versions after v9y** (details in `submissions/LOG.md`): v9zm adds 5,883 France recoveries and a US/India
> candidate-list fix (public 0.989968); v10a blends our US/India probability 0.6/0.4 with the second pipeline's
> (public 0.990166); v10b adds 4,825 France matches from the second generator (`src/france_recall.py`). The steps
> below rebuild v9y exactly; the later blends are being moved into `src/` for the Round 2 package.

- `matching_results.tsv`: one row per test Source 1 (S1) entity, the matched Source 2 / Source 3 records.
- `candidate_pairs.tsv`: the pairs the final matching models score (definition in step 19).
- Check done on 27 Sep 2026: steps 16, 18 and 19, run from the saved intermediate files (step 17's output file as
  received), rebuilt both submitted files byte for byte (md5 `2ebc84acd2cef360a9296ad6ee587b8c` for
  `matching_results.tsv`, `de40e20b1f9d975a4ee5307bf60324e3` for `candidate_pairs.tsv`); the official
  `validate_submission.py --check-ids` printed PASS on them.

## 1. Where each part ran

| Machine | Hardware | What ran there |
|---|---|---|
| Team laptop (captain) | Intel i5-12450H, 16 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11, Python 3.12.10 | every step except the three below |
| Friend 2's workstation | NVIDIA RTX A6000 48 GB, Ubuntu | step 12 (bge-reranker-v2-m3, 3 fold models: training and scoring) and step 14b (bge scores for the new pairs of the wide search) |
| Lab GPU server (shared; run by teammate Gathik) | NVIDIA RTX PRO 6000 Blackwell 96 GB, Rocky Linux 8 | step 17 (`server/run_v9.sh`: the same pipeline with the wider `BER_V8=1` blocking, as a second candidate generator and matcher; 16,421 of its US/India pairs are in the final file) |
| Friend 1's laptop | NVIDIA RTX 4060 8 GB | multilingual-e5-base 3-fold run; used in earlier versions (v7g), **not** in the final file |

Files moved between machines by SSH/SFTP (`scp` works the same). The helper script used for this is not included
because it holds login credentials.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # pins the laptop's versions, incl. torch 2.7.1+cu118
```

- The laptop GPU driver supports CUDA 11.x only, hence `torch==2.7.1+cu118` and `xgboost==2.0.3`.
  On a newer driver, a CUDA 12 build of torch and a newer xgboost work with the same code.
- A CUDA GPU is required: XGBoost stages run with `device="cuda"`, and the transformers train and score on the GPU.
- Disk: about 90 GB for the work folder (train pair features 18 GB, test 18 GB, wide test build 25 GB, transformer
  files ~10 GB, embeddings 2.6 GB). RAM: 16 GB is enough when steps run one at a time.

## 3. Folders and environment variables

All commands run from `src/`. Three environment variables set the folders (defaults in `src/common.py`):

```bash
cd code/business_entity_resolution/src
export BER_DATA=/path/to/student_resource/dataset   # holds train/ and test/ with the six *_source{1,2,3}.tsv + train_ground_truth.tsv
export BER_WORK=/path/to/work                        # every intermediate file
export BER_OUT=/path/to/output                       # the two final TSV files
export PYTHONIOENCODING=utf8
W=$BER_WORK
```

- On Windows (Git Bash) use drive paths such as `E:/ber/work` in these variables; Python does not read `/e/...` paths.
- Use folder paths without spaces: the commands below leave `$W` and `$BER_OUT` unquoted, so a path with a space
  is split into several arguments (quote them, e.g. `"$W/..."`, if the path must contain spaces).
- If unset: `BER_DATA=<package root>/student_resource/dataset`, `BER_WORK=<package root>/work`,
  `BER_OUT=<package root>/output`.
- Every long step skips work already on disk (chunk files, checkpoints, saved score chunks), so a stopped step is
  resumed by running the same command again.

## 4. Pipeline in one picture

```
raw TSV -> cleaning (1) -> e5 embedding for non-Latin names (2)
        -> blocking + 55 pair features, train and test (3, 5, 13)
        -> stage 1 LightGBM (4): scores every blocked pair, keeps the top candidates of each record
        -> stage 2 XGBoost with group features (4) -> p2 per pair
        -> close calls (uncertain records) + stage-1 ranks 3-5 (7)
        -> cross-encoders: e5-small and bge-reranker-v2-m3, 3 folds each (11, 12, 14)
        -> stage 3 XGBoost per family, blend 0.3/0.7 (15) -> US/India rows
France:    e5-small/e5-base halves transformers (8-10) + France rules (16) -> France rows
Gathik:    same pipeline with BER_V8=1 blocking (17) -> extra US/India pairs for records our file leaves unmatched
assemble (18) -> matching_results.tsv;  make_candidates (19) -> candidate_pairs.tsv
```

## 5. Commands, in order

### Step 1. Cleaning (laptop CPU, minutes)

```bash
python learn_maps.py     # address short forms (rd -> road) and local-script state names, learned from TRAIN matches -> $W/maps.json
python prep.py           # clean all six source files -> $W/{train,test}_s{1,2,3}.parquet
```
Do not set `BER_V8` for these steps (it switches on the cleaning of step 17, which is a different run).

### Step 2. Embedding model for non-Latin names (laptop GPU, ~10 min)

```bash
python embed.py train    # fine-tunes intfloat/multilingual-e5-small on (non-Latin record, S1) train pairs -> $W/e5_ft_addr/
python embed.py encode   # vectors for S1 rows of countries with non-Latin records + those records -> $W/emb/{train,test}_{s1,q}.npz
```

### Step 3. Blocking and pair features for all of TRAIN (laptop, ~2.4 h)

```bash
python pipeline.py build train full          # -> $W/pairs/full/<country>_<chunk>.parquet (319.0M pairs, 30.9 per record)
```

### Step 4. Stage 1 and stage 2 (laptop; stage 1 CPU ~1.7 h, stage 2 GPU ~13 min)

```bash
BER_BACKEND=lgb python pipeline.py train full                        # stage 1 LightGBM -> $W/models/full/ (s1_f0..2, s1_full, OOF table)
BER_BACKEND=lgb BER_BACKEND2=xgb python pipeline.py train full cons  # stage 2 XGBoost with sibling + consensus features -> $W/models/full_cons/
```
`BER_BACKEND=lgb` matters: the default of `pipeline.py` is XGBoost for stage 1, while the final models use LightGBM.
The first command also trains a LightGBM stage 2 that later steps do not use; it must still run, because
`topk5.py` (step 7) reads its `$W/models/full/result.json`.

### Step 5. Blocking for TEST (production settings) and test scores (laptop, ~1.5 h + ~1.3 h)

```bash
python pipeline.py build test test                                 # -> $W/pairs/test/ (307.2M pairs)
BER_OUT=$W/out_full_cons python pipeline.py predict full_cons test # -> $W/test_scores_full_cons.parquet (q, s, p1, p2)
```
`predict` also writes an intermediate submission into `$W/out_full_cons/`; it is not used. The
`candidate_pairs.tsv` written there holds every searched pair (307.2M, the raw blocking output before the stage-1
filter); it is not the submitted candidate file, which step 19 writes.

### Step 6. Close calls (laptop, seconds)

```bash
python rerank.py select full_cons    # -> $W/ce/{train,test}_rows.parquet, $W/ce/select.json
```
A close call is a record whose best p2 is in [0.01, 0.995] or whose 2nd candidate has p1 >= 0.2; its top 2
candidates are rescored by the transformers.

### Step 7. Stage-1 candidates ranked 3-5 of the close-call records (laptop, ~45 min)

```bash
python topk5.py train full    # -> $W/models/full/stage1_rank3_5.parquet (out-of-fold p1)
python topk5.py test full     # -> $W/test_rank3_5_test_full.parquet
BER_CE_DIR=$W/ce_x python rerank.py extras full_cons $W/ce    # close calls + ranks 3-5 -> $W/ce_x/{train,test}_rows.parquet
```
`topk5.py` reads free RAM through a Windows API call (`free_gb`); on Linux replace that function with a constant.

### Steps 8-10. Transformers used for the France rows (laptop GPU, ~7 h in total)

These three cross-encoders were trained on the two halves of train S1 (side `a`: the half not used for evaluation;
side `b`: the evaluation half; side `a2`: a larger model on side a's rows). Their blend decides the France rows
(France has no labels; step 16 explains why France keeps this older setting).

```bash
# step 8: e5-small, side a (~1-1.5 h training + ~1 h scoring) and side b (~50 min + ~35 min)
python rerank.py train intfloat/multilingual-e5-small              # -> $W/ce/model/
python rerank.py score                                             # -> $W/ce/{train,test}_ce.parquet
BER_CE_SIDE=b python rerank.py train intfloat/multilingual-e5-small
BER_CE_SIDE=b python rerank.py score                               # -> $W/ce/{train,test}_ce_b.parquet
python rerank.py stage3                                            # -> $W/ce/test_scores_ce_ab.parquet, oof_s3_ab, rule_ab.json
# step 9: e5-base, side a2, 420k training rows (16-bit frozen word table and batch 16 to fit 4 GB; ~1.4 h + ~1.4 h)
BER_CE_SIDE=a2 BER_CE_BS=16 BER_CE_HALF_EMB=1 BER_CE_LIMIT=420000 python rerank.py train intfloat/multilingual-e5-base
BER_CE_SIDE=a2 python rerank.py score
BER_S3_SIDES=a2 python rerank.py stage3                            # -> $W/ce/test_scores_ce_a2.parquet
# step 10: blend (weights chosen on held-out) and France min rule
python blend.py ab_a2 full_cons:ce:_ab full_cons:ce:_a2            # -> $W/test_scores_blend_ab_a2.parquet, $W/ce/rule_blend_ab_a2.json
python fr_minrule.py $W/test_scores_blend_ab_a2.parquet $W/test_scores_full_cons.parquet $W/ce/test_scores_blend_ab_a2_frmin.parquet
```

### Step 11. e5-small, 3 fold models (laptop GPU, ~6 h)

```bash
python ce_folds.py --model intfloat/multilingual-e5-small --name small
# -> models $W/ce/model_smallf{0,1,2}/, scores $W/ce_x/{train,test}_ce_smallf{0,1,2}.parquet
```
Fold model k trains on every labelled close call whose record is in another pipeline fold (1.68M pairs), then
scores fold k (out-of-fold) and all test close calls. On a 4 GB GPU the word-embedding table is frozen, gradient
checkpointing is on, batch 32, learning rate 3e-5, 1 epoch, 128 tokens.

### Step 12. bge-reranker-v2-m3, 3 fold models (friend 2's RTX A6000 48 GB, ~10-12 h)

On the laptop, pack the inputs (names/addresses of close-call records only, 329 MB; no raw files):
```bash
python bundle.py      # -> $W/bundle_ce_x.zip
```
On the GPU machine (repo copy, `bundle_ce_x.zip` unzipped in the package root so that `work/` is created there):
```bash
cd code/business_entity_resolution/src
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --smoke --no-freeze --batch 32 --lr 2e-5 --score-batch 512   # 3-min check
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
```
Full fine-tuning (word table trained), batch 64, learning rate 2e-5, 1 epoch (26,000 steps per fold), about 3.4 h
per fold for training plus scoring (fold 2: 05:56 to 09:18 on 27 Sep, `work/f2_watch.log`); the card was shared with
another user's job for part of the run. Copy back into `$W/ce_x/`: `train_ce_bgef{0,1,2}.parquet`, `test_ce_bgef{0,1,2}.parquet`.
Keep the three models (`work/ce/model_bgef{0,1,2}/`) on the GPU machine for step 14b.

### Step 13. Wide test blocking for US and India (laptop, ~3.6 h + ~1.5 h)

```bash
python blocking_b2.py build      # combined name+address search max_df 20000, top 40 (production: 5000, top 20)
                                 # -> $W/pairs/test_b2/ (469.8M pairs; France chunks hard-linked from pairs/test)
python blocking_b2.py predict    # stage 1 + ranks 3-5 + stage 2 of full_cons -> $W/test_scores_full_cons_test_b2.parquet,
                                 #   $W/test_rank3_5_test_b2.parquet (prints a check: France p2 equals step 5 exactly)
python blocking_b2.py rows       # -> $W/ce_b2/test_rows.parquet (3,788,098 rows); train rows/scores linked from ce_x
```

### Step 14. Transformer scores for the new pairs of the wide search

Pairs already scored in `ce_x` are copied; only the 585,295 new pairs are scored.

14a, laptop (e5-small, ~5 min per fold):
```bash
for k in 0 1 2; do
  BER_CE_DIR=$W/ce_b2 BER_CE_SIDE=f$k BER_CE_NAME=small BER_CE_REUSE=$W/ce_x python rerank.py score
done
```
14b, friend 2's machine (bge, ~5 min per fold). It needs the **full** cleaned test files
`work/test_s{1,2,3}.parquet` from step 1 (the bundle of step 12 holds cut-down copies; scoring with those gives empty
texts) and `work/ce_b2/test_rows.parquet` from step 13:
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
                                       # weights 0.3 / 0.7 chosen on held-out -> $W/test_scores_blend_v7p.parquet, $W/ce/rule_blend_v7p.json
python fr_minrule.py $W/test_scores_blend_v7p.parquet $W/test_scores_full_cons.parquet $W/ce/test_scores_blend_v7p_frmin.parquet
BER_CALIB=$W/ce/rule_blend_v7p.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons $W/out_v7p_all $W/ce/test_scores_blend_v7p_frmin.parquet
```
Only the US and India rows of `$W/out_v7p_all/matching_results.tsv` are used.

### Step 16. France rows (laptop, ~1 min)

The France rows start from the scores of step 10 and apply three pair sets, in this order:

| Set | Pairs | Change | Defined by (analysis scripts in `france_fix/`) |
|---|---|---|---|
| `desc_veto_set.parquet` | 22,436 | p2 = 0 | `namechg/a1_build.py` ... `a11_veto.py`: accepted France pairs whose record adds or swaps in a descriptor word (amicale, comite, ecole, club, centre, ...), synonym and stem swaps excluded |
| `noise_add_set.parquet` | 9,800 | p2 = max(p2, 0.95) | `build/b4_addset.py`, `build/b8_mkscores_v2.py`: rejected same-address pairs whose added or swapped word is a noise word, or groupe/developpement/france where the v7m probe had restored the pair |
| `fp_veto_set.parquet` | 1,145 | p2 = 0 | `artifacts/fp/fp1_dec.py` ... `fp10_veto.py`: accepted pairs in three groups (column `tier`): A, 846 pairs, legal form added and house number moved up by 1-20 (decoy signature); B, 232 pairs, word added, swapped or mistyped and house number moved up by 1-20; C, 67 pairs, all-lowercase record with a name change |

The three sets ship in `france_fix/sets/` (parquet files: in the submission zip, not in git). Apply them and decide the France rows with the rule of step 10:
```bash
FS=../france_fix/sets SC=$W/ce/test_scores_blend_ab_a2_frmin.parquet DST=$W/test_scores_france_final.parquet python - <<'EOF'
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
Only the France rows of `$W/out_france/matching_results.tsv` are used. `BER_FR_LEGAL_VETO=1` also sets p2 = 0 for
France pairs whose two raw names both carry a legal form (SARL, SAS, SA, EURL, ...) with none in common.

To re-derive the three sets instead of using the shipped files: run the scripts named in the table in file-name
order. They were run once, interactively, on the laptop; each has its input/output folders as constants at the top
of the file (`FR`, `OUT`, `NC`, `RD`, `F`, `T`, `W`), which must be pointed at a scratch folder first. Their inputs
are best-candidate tables (`fr_top.parquet` for France test records, `usi_top.parquet` for labelled US/India train
records, `pairs_<version>.parquet` for the France pairs of earlier submissions) built by `france_fix/tables/fr1.py`,
`fr4.py` and `fr8.py` from the step 5 and step 10 test scores, the out-of-fold train scores
(`$W/models/full_cons/oof.parquet`, `$W/ce_x/oof_s3_smallfolds.parquet`) and earlier submission files (v7ens and the
France probe versions listed in the methodology document).

### Step 17. Second candidate generator and matcher (lab GPU server, run by Gathik; `server/run_v9.sh`)

A second full run of this pipeline with the audited blocking (`BER_V8=1`: name search max_df 10000, no-address name
top 100, word searches with min_df 1, dense e5 top-10 search for every record, India state-code and ordinal
cleaning), an XGBoost stage 1 and 3 bge-reranker-v2-m3 fold models. It needs its own work folder because its
cleaning differs from step 1.

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
`server/run_v9.sh` is the scripted version (with a smoke test, retries and waits for free RAM on the shared server).
Copy `$BER_OUT/matching_results.tsv` to the laptop as `$W/gathik/v8_matching_results.tsv` (the file name is
historical; its per-country counts, US 2,255,299, India 2,748,510, France 853,628 matched pairs, are those of this run
in `EXPERIMENTS.md`). Only its US/India rows are used. Held-out macro F0.5 of this run: 0.99183 (all close calls
rescored). The transformer stage ran 27 Sep 10:27-12:54 IST; the full server run time was not recorded.

`src/v8.py` is an earlier streamed build on the same server (files v8a/v8b); it is not part of the final file.

### Step 18. Final matching file (laptop, seconds)

```bash
python assemble_final.py $W/out_v7p_all/matching_results.tsv $W/out_france/matching_results.tsv $BER_OUT \
    $W/gathik/v8_matching_results.tsv $W/test_scores_blend_v7p.parquet
```
US/India rows from step 15, France rows from step 16, plus the step 17 pairs whose (record, S1) pair is absent from
our candidate lists, for records our file leaves unmatched (16,421 pairs: India 12,939, US 3,482).
Output: `$BER_OUT/matching_results.tsv` (5,846,292 matched pairs; 1,632,486 of 1,732,544 S1 rows non-empty).

### Step 19. Candidate file (laptop, seconds)

```bash
python make_candidates.py $W/test_scores_blend_v7p.parquet $BER_OUT/matching_results.tsv $BER_OUT \
    $W/gathik/v8_matching_results.tsv
```
`candidate_pairs.tsv` = every (record, S1) pair of the final stage-2/stage-3 score table: each record's top stage-1
candidates (the input of stage 2) plus the stage-1 ranks 3-5 of close-call records (scored by the transformers),
plus the 16,421 step 17 pairs used in step 18. The script stops with an error if any matched pair is not a
candidate. Result: 11,658,294 pairs, 6.73 per S1 row (US 6.29, India 6.70, France 7.94), 377 S1 rows without
candidates.

### Step 20. Checks

```bash
python check_submission.py $BER_OUT/matching_results.tsv             # fast check of every leaderboard rule
cd <student_resource folder>
python utils/validate_submission.py --matching <BER_OUT>/matching_results.tsv \
    --candidate <BER_OUT>/candidate_pairs.tsv --test-dir dataset/test --check-ids   # official validator: PASS
```

## 6. The e5-large family (tested, not used)

A third cross-encoder family (intfloat/multilingual-e5-large, MIT, ~560M parameters, published figure) was trained on
27 Sep (folds 0-1 on friend 2's machine, fold 2 on the lab server). Held-out, all close calls rescored: e5-large alone
0.98924; three families equal 0.98946; bge 0.7 + e5-large 0.3 0.98954; v7p (e5-small 0.3 + bge 0.7) 0.98950. The gain
is within noise, so the final file does not use it. It would enter between steps 14 and 15:

```bash
# friend 2's machine, as step 12 and 14b with --name e5l:
python ce_folds.py --model intfloat/multilingual-e5-large --name e5l --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
# (then step 14b with BER_CE_NAME=e5l, BER_CE_MODEL=.../model_e5lf$k, output ce_b2/test_ce_e5lf$k.parquet)
# laptop: copy train_ce_e5lf{0,1,2} into $W/ce_x and $W/ce_b2, test_ce_e5lf{0,1,2} into $W/ce_b2, then
BER_CE_DIR=$W/ce_b2 BER_S3_EXTRA=1 BER_S3_SIDES=e5lf0,e5lf1,e5lf2 BER_S3_TAG=_e5lfolds \
  BER_S3_TEST=$W/test_scores_full_cons_test_b2.parquet python rerank.py stage3
python blend.py v7r full_cons:ce_b2:_smallfolds full_cons:ce_b2:_bgefolds full_cons:ce_b2:_e5lfolds   # 3 families: equal weights
```
Steps 15 (from `fr_minrule.py` on), 18 and 19 then use `v7r` in place of `v7p`. The scripted version of this is
`overnight/e5l_after.sh` and `overnight/b2_e5l_remote.sh` (GPU machine) and `overnight/e5l_final.sh` (laptop).

## 7. Limits of this reproduction

- GPU training is not bit-for-bit deterministic; retrained transformers give slightly different scores, so a full
  rerun gives a file close to, not identical with, the submitted one. Steps 16, 18 and 19 are exact given the saved
  scores and the step 17 file.
- Steps 12, 14b and 17 ran on machines other than the laptop (section 1). The step 17 output was received as a file.
- The France pair sets (step 16) were chosen with label-free tests and France-only leaderboard submissions (see the
  methodology document); re-deriving them needs the analysis scripts in `france_fix/`, which were written for one
  interactive run and use hard-coded folders.
- `overnight/*.sh` are the shell scripts that ran steps 13-15 unattended on 27 Sep; they contain the laptop's
  folder names and call the SSH helper that is not included.

## 8. Source files

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

Other folders: `server/` holds the lab-server drivers (`run_v9.sh` = step 17, started by `v9_after_v8.sh`;
`run_v8.sh`, `run_v8_resume.sh`, `v8_guard.sh` for the earlier `v8.py` build; `watchdog.sh` stops a test build when
free RAM runs low), `overnight/` the laptop and GPU-machine scripts of 27 Sep, and
`build_combo.sh` builds a submission from 3-fold transformer families (used for v7f/v7g/v7q, not for the final file).

Experiments and analysis, not needed to rebuild the files: `v8.py` (earlier streamed build on the lab server) and
`dl_matcher_exp.py` (transformer as final matcher; `v8.py` imports its cross-encoder class), `avg_family.py`,
`blocking_exp.py`, `blocking_variants.py`,
`calib.py`, `cmp_fold0.py`, `combined_blocking_exp.py`, `decoy_sim.py`, `dense_blocking_exp.py`, `dlx_analyse.py`,
`embed_all.py`, `error_analysis.py`, `error2.py`, `evaluate.py`, `fr_desc.py`, `fr_num.py`, `hybrid_b2.py`,
`merge_country.py`, `name_cap_exp.py`, `name_search_diag.py`, `ordinal_exp.py`, `selftrain.py`, `splice_country.py`,
`stage3_joint.py`, `testlike.py`, `testlike_build.py`. `france_cal.py` and `fr_restore.py` are imported by the
`france_fix/` analysis scripts.

## 9. Models and licences

| Model | Licence | Parameters | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 117.7M (counted from the saved weights) | embedding search (fine-tuned); cross-encoders A, B and 3 folds |
| intfloat/multilingual-e5-base | MIT | 278.0M (counted) | cross-encoder A2 (France rows) |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.8M (counted) | cross-encoder, 3 folds (US/India); cross-encoder of the step 17 run |
| intfloat/multilingual-e5-large (only if section 6 is used) | MIT | ~560M | cross-encoder, 3 folds |
| LightGBM 4.7.0 / XGBoost 2.0.3 | MIT / Apache-2.0 | tree ensembles | stages 1-3 |

The largest model has under 0.6B parameters (limit: 8B). No external data, APIs or lookup services are used; the
learned tables and tree models are fitted on the provided training data, and the transformers are the public
checkpoints above fine-tuned on the provided training data.
