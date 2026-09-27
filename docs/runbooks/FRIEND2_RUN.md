# GPU run on the RTX A6000 (48 GB, Ubuntu) — friend 2, 26–27 Sep

**What this run does:** trains 3 models of **BAAI/bge-reranker-v2-m3** (Apache-2.0, 568M parameters, a multilingual model already trained to judge whether two texts match) on our labelled record pairs, and scores every uncertain pair of the test set with them. These scores replace/join the transformer scores in our submission.

**Why this model on this GPU:** the transformer re-check is our biggest gain (leaderboard 0.9700 → 0.983159). A same-size model with this run's design already beat our old one on 280k held-out pairs (AUC 0.9911 vs 0.9887). The A6000 can train a model ~5× larger than ours, fully (no frozen parts), which the 4–8 GB GPUs cannot. Friend 1 (RTX 4060) runs the mid-size model (e5-base) at the same time; different models, no duplicated work.

**Time:** roughly 1.5–2.5 h per model, 5–7 h for all 3 (the training log prints the real speed in the first minutes).

---

## 1. Setup (once)

```bash
git clone https://github.com/Paradox-73/Business_Entity_Resolution.git
cd Business_Entity_Resolution
python3 -m venv .venv && source .venv/bin/activate
pip install torch                       # Linux wheels include CUDA 12 (driver 570 / CUDA 12.8 is fine)
pip install "transformers>=4.44" polars pyarrow numpy sentencepiece protobuf scikit-learn
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, NVIDIA RTX A6000
```

Kanav sends `bundle_ce_x.zip` (329 MB; data, so never in git). Unzip it **in the repo root**:

```bash
unzip bundle_ce_x.zip        # creates work/train_s1.parquet ... work/ce_x/train_rows.parquet
```

Needs ~5 GB of RAM, ~10 GB of disk (model download 2.2 GB + outputs). Run inside `tmux` or `screen` so a closed terminal does not stop it:

```bash
tmux new -s ber            # re-attach later with: tmux attach -t ber
```

## 2. Smoke test (~3 minutes)

```bash
cd code/business_entity_resolution/src
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --smoke --no-freeze --batch 32 --lr 2e-5 --score-batch 512
```

It must end with `DONE. Send back: …_smoke.parquet`. If not, send Kanav `work/ce_folds_bge_smoke.log` and `work/ce_folds_bge_*.log`.

## 3. Full run

```bash
python ce_folds.py --model BAAI/bge-reranker-v2-m3 --name bge --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
```

- Progress: `work/ce_folds_bge.log`; speed: `work/ce_folds_bge_train_fold_0.log` prints `step N/26000` every 500 steps (26,000 steps per model at batch 64).
- If one model's training would take more than 2.5 h (e.g. under 3 steps/s), stop with Ctrl-C and restart with `--train-rows 1000000` added; finished models are kept.
- Out of GPU memory (unlikely on 48 GB): the step retries 3 times by itself; if it still fails, use `--batch 32`.
- If the machine restarts or the command stops: run the same command again. Finished models and scored chunks are skipped; training resumes from its last checkpoint (every 1,000 steps).

## 4. Send back — after EACH model, not only at the end

When the log says `OK score fold 0`, send these two files right away (Kanav checks model 0 while models 1 and 2 train):

```
work/ce_x/train_ce_bgef0.parquet   work/ce_x/test_ce_bgef0.parquet
```

then the same for `bgef1` and `bgef2`. About 150 MB per model. Do not commit anything to git.

## 5. Only if all 3 are done and there is time before Sun 27 Sep 16:00

A second large model for the blend:

```bash
python ce_folds.py --model intfloat/multilingual-e5-large --name large --no-freeze --batch 64 --lr 2e-5 --score-batch 1024
```

Send back the `largef0..2` files the same way.

---

## What Kanav's laptop does with the files

```bash
cd code/business_entity_resolution/src
python cmp_fold0.py bge          # model 0 vs the old transformer on the same held-out pairs
BER_CE_DIR=../../../work/ce_x BER_S3_EXTRA=1 BER_S3_SIDES=bgef0,bgef1,bgef2 BER_S3_TAG=_bgefolds python rerank.py stage3
python blend.py final full_cons:ce_x:_bgefolds full_cons:ce_x:_basefolds full_cons:ce_x:_smallfolds
# France keeps the setting validated on the leaderboard (transformer may only remove France matches), then finalize.py
```
A version is uploaded only if its held-out ("halves protocol" line) beats v7ens's 0.98813.
