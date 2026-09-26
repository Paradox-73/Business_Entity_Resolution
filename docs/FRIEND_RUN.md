# GPU run on the RTX 4060 laptop (26–27 Sep)

**What this run does:** trains 3 transformer models (multilingual-e5-base, MIT licence) that re-check the "close calls" of our best model, and scores every close call with them. It needs a CUDA GPU with 8 GB. It does not need the competition data: a 329 MB bundle has only the names and addresses involved.

**Why:** the transformer re-check is our biggest gain so far (leaderboard 0.9700 → 0.983159). Until now it was trained on half the labels with a small model on a 4 GB GPU. Here each model sees 2/3 of all labels (1.68M pairs instead of 0.83M), and the model is 2.4× bigger.

**Time:** about 1.8 h per model with `--train-rows 800000` (5–6 h for all 3; 8–9 h without the cap) (estimate from the 4 GB laptop: e5-base trained at 90 pairs/s and scored 714 pairs/s there). The first training steps print the real speed.

---

## 1. Setup (once)

Requirements: Windows or Linux, NVIDIA driver for the RTX 4060, Python 3.10–3.12, Git, ~6 GB free disk, internet (downloads the 1.1 GB model once).

```bash
git clone https://github.com/Paradox-73/Business_Entity_Resolution.git
cd Business_Entity_Resolution
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.44" polars pyarrow numpy sentencepiece protobuf
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True and the RTX 4060
```

Kavya sends `bundle_ce_x.zip` (329 MB) separately — it is data, so it is never in git. Unzip it **in the repo root** (it creates `work/…`):

```bash
# Windows PowerShell:  Expand-Archive bundle_ce_x.zip -DestinationPath .
# Linux / Git Bash:    unzip bundle_ce_x.zip
```

Check: `work/train_s1.parquet`, `work/test_s3.parquet`, `work/ce_x/train_rows.parquet` exist.

Before starting: plug in the charger, set the power mode to "Best performance", close browsers and games (the run needs ~4 GB of RAM and the whole GPU). Do not run anything else on the GPU during the run.

## 2. Smoke test (3 minutes)

```bash
cd code/business_entity_resolution/src
python ce_folds.py --model intfloat/multilingual-e5-base --name base --smoke
```

It must end with `DONE. Send back: …_smoke.parquet`. If it fails, send Kavya the file `work/ce_folds_base_smoke.log` and the logs `work/ce_folds_base_*.log`.

## 3. Full run (~5–6 h)

```bash
python ce_folds.py --model intfloat/multilingual-e5-base --name base --train-rows 800000
```

`--train-rows 800000` (26 Sep 16:00): each model trains on 800k of its 1.68M pairs (still ~1.6x more than before) so the GPU is free by Sunday morning for a France job. If this run is already more than ~1 h in without the flag, let it finish instead of restarting.

- Progress: `work/ce_folds_base.log` (one line per step) and `work/ce_folds_base_train_fold_0.log` (a line every 500 training steps, e.g. `step 500/52500`).
- If the GPU runs out of memory, the step is retried up to 3 times automatically. If it still fails, add `--gc --half-emb` (less memory, ~30% slower), then `--batch 16`.
- If the laptop restarts or the command is stopped: run the same command again. Finished models and scored chunks are kept; training resumes from its last checkpoint (every 1,000 steps).
- Laptop sleep stops the run: set sleep to "never" while plugged in.

## 4. Send back (6 files, ~0.5 GB)

```
work/ce_x/train_ce_basef0.parquet   work/ce_x/test_ce_basef0.parquet
work/ce_x/train_ce_basef1.parquet   work/ce_x/test_ce_basef1.parquet
work/ce_x/train_ce_basef2.parquet   work/ce_x/test_ce_basef2.parquet
```

Send each fold's two files as soon as that fold is done (the log says `OK score fold k`), so Kavya can start checking early. Do not commit anything to git.

## 5. Only if the base run finished and more than 10 h remain before Sun 27 Sep 18:00

A larger model on fewer rows (speed unknown on this GPU; stop it if a fold would take more than 5 h):

```bash
python ce_folds.py --model intfloat/multilingual-e5-large --name large --batch 16 --train-rows 800000 --gc --half-emb
```

Send back the same 6 files with `large` instead of `base`.

---

## What Kavya's laptop does with the files (for reference)

```bash
cd code/business_entity_resolution/src
# stage 3 on the 3 fold models (prints held-out; "halves protocol" is comparable with v7ens 0.98813)
BER_CE_DIR=../../../work/ce_x BER_S3_EXTRA=1 BER_S3_SIDES=basef0,basef1,basef2 BER_S3_TAG=_basefolds python rerank.py stage3
# blend with the laptop's e5-small folds (v7f), France rule, submission file v7g
python blend.py final full_cons:ce_x:_basefolds full_cons:ce_x:_smallfolds
python fr_minrule.py ../../../work/test_scores_blend_final.parquet ../../../work/test_scores_full_cons.parquet ../../../work/ce/test_scores_blend_final_frmin.parquet
BER_CALIB=../../../work/ce/rule_blend_final.json BER_FR_LEGAL_VETO=1 python finalize.py full_cons ../../../work/out_v7g ../../../work/ce/test_scores_blend_final_frmin.parquet
python check_submission.py ../../../work/out_v7g/matching_results.tsv
```
A version is uploaded only if its held-out (halves protocol) beats v7ens's 0.98813.
