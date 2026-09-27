# Lab GPU server runs (Gathik, 26–27 Sep)

**What ran there:** the deep-learning matcher experiment (`dl_matcher_exp.py`, results in EXPERIMENTS.md) and the overnight v8 submission build. Written so a teammate can reproduce or continue the runs.

## Machine and house rules
- Shared Linux server (Rocky Linux 8), reached with `ssh gpu` (host alias in `~/.ssh/config`; ask Gathik for access).
- GPUs 0 and 1 (RTX 6000 Ada 48 GB) belong to other users' jobs. **Use only GPU 2** (RTX PRO 6000 Blackwell, 96 GB): every command runs with `CUDA_VISIBLE_DEVICES=2`.
- `/home` is shared and nearly full (it reached 100% once; ~20 GB free since). **Never store full pair-feature files there**: they need ~57 GB for train + test. Stream instead (score each chunk, keep the top candidates).
- RAM is shared too (215 GB, often only 20–40 GB free). Build chunks of 125k records (`BER_CHUNK=125000`), threads capped (`BER_THREADS=24`).

## Setup (already done under `~/ber`)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv in ~/.local/bin
uv venv --python 3.12 ~/ber/.venv && . ~/ber/.venv/bin/activate
uv pip install torch==2.8.0                              # PyPI build = CUDA 12.8, needed for the Blackwell GPU (sm_120)
uv pip install polars==1.44.2 numpy scipy scikit-learn lightgbm==4.7.0 rapidfuzz==3.14.6 sparse_dot_topn==1.2.0 \
    pyarrow transformers==5.17.0 xgboost sentencepiece protobuf
```
- `pypi.nvidia.com` times out from this machine, so the `download.pytorch.org` cu128 wheels do not install; the PyPI torch 2.8 wheel does.
- Layout: `~/ber/repo` (code + raw dataset), `~/ber/w` (`BER_WORK`: cleaned parquet with the v8 cleaning, `emb/` all-record e5 embeddings, experiment outputs), `~/ber/out` (`BER_OUT`).
- Copying data from a laptop over Wi-Fi/VPN runs at ~2–5 MB/s: send the raw TSVs gzip-compressed and rebuild the parquet on the server (`prep.py` takes 30 s there). Verify every copied file (md5) before using it.

## Commands
```bash
export BER_V8=1 BER_WORK=~/ber/w BER_DATA=~/ber/repo/student_resource/dataset BER_OUT=~/ber/out
export CUDA_VISIBLE_DEVICES=2 BER_THREADS=24 POLARS_MAX_THREADS=24 BER_CHUNK=125000
cd ~/ber/repo/code/business_entity_resolution/src
python prep.py                                   # v8 cleaning -> $BER_WORK/*.parquet
python embed.py encode_all                       # e5 vectors for every S1 and S2/S3 row (~40 min for train + test)
BER_DLX_S1=40000 BER_DLX_TRAIN=240000 python dl_matcher_exp.py   # ~45 min; then python dlx_analyse.py
```
Long jobs run inside `tmux` (`tmux new -s <name>`), so a dropped ssh connection does not stop them.

## v8 / v9 submission builds (drivers in `code/business_entity_resolution/server/`, copied to `~/ber` on the server)
- `run_v8.sh`: smoke test, then `v8.py train | gbdt | test | ce | final` with retries -> `~/ber/out/v8a`, `~/ber/out/v8b` (France variants). `run_v8_resume.sh` + `v8_guard.sh` resume it (10 tries per step) whenever the driver dies before `w/v8/summary.json` exists.
- `run_v9.sh`: FULL rebuild with the v8 blocking (train + test pairs on disk, ~60 GB), `pipeline.py train full cons`, `predict`, `rerank.py select`, `ce_folds.py` (bge-reranker-v2-m3, 1M rows per fold), `rerank.py stage3`, `fr_minrule.py`, `finalize.py` (legal veto) -> `~/ber/out/v9/` incl. candidate_pairs.tsv. `v9_after_v8.sh` starts it only after v8 finished (one heavy job at a time).
- `watchdog.sh`: stops our (restartable) test build if free RAM drops below 6 GB, so the kernel OOM killer never picks another user's job.
- Check progress: `tail -f ~/ber/v8.log ~/ber/v9.log`; sessions: `tmux ls`.
