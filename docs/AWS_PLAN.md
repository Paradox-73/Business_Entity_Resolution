# AWS run plan — Saturday 26 Sep (and Sunday 27 Sep)

## Rules check (Amazon forum answers, 25 Sep)
- Allowed: MIT/Apache-2.0 open models ≤ 8B each, offline, fine-tuned only on provided data; pure-algorithm libraries (RapidFuzz, scikit-learn, LightGBM, XGBoost, pandas); small hand-written normalisation dictionaries; unsupervised statistics on the test files (TF-IDF, blocking index); self-training and synthetic pairs from provided records.
- Not allowed: external data, APIs, geocoders, libpostal, postal/gazetteer datasets, hosted LLM APIs.
- Everything below complies: mDeBERTa-v3 (MIT) and multilingual-e5 (MIT), fine-tuned on the provided train pairs only.

## Where we stand (end of 25 Sep)
- LB: v2 0.9677, **v3 0.9700 (best)**, v5 0.838 (France probe), v4 0.9614, v6 0.9673. Leader ~0.984–0.987.
- From v5: France ≈ 0.937, US+India ≈ 0.976.
- Held-out (FULL validation): v3 0.98015 → v6 base 0.98151, yet v6's LB fell. Held-out gains failed to reach the LB twice (v4, v6) → **the validation does not look like test.** A test-like validation built with the real search (`tl2`) is being built on the laptop tonight.
- Where held-out points go (US/India, v2 error analysis): wrong S1 chosen 0.009 (65% of those records have no address), right S1 below the cutoff 0.008, true pair never shortlisted 0.005 (recall ceiling 0.985), look-alike merged 0.005.

## What 0.99 would take (honest)
- LB = 0.85 × US/India + 0.15 × France. 0.99 needs both ≈ 0.99: US/India +0.014 and France +0.05 on the LB.
- The two AWS jobs target the biggest measured buckets (reranker → wrong S1 / below cutoff; retriever → never shortlisted). Estimated LB by Sunday if both work: **0.975–0.98**. Neither is expected to reach 0.99 alone; the gates below decide what is kept.

## Machines
| Machine | Instance | Use | Quota to request |
|---|---|---|---|
| GPU | **g5.2xlarge** (1× A10G 24 GB, 8 vCPU, 32 GB RAM), 200 GB gp3 | G1 reranker, G2 retriever, stage-2 training | "Running On-Demand G and VT instances" ≥ 8 vCPU |
| CPU | **c7i.16xlarge** (64 vCPU, 128 GB RAM), 300 GB gp3 — only from job C1 on | rebuild candidates + features for train and test with the new search (CPU-bound string features; ~6.5 h on the laptop) | "Running On-Demand Standard instances" ≥ 64 vCPU |

AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu) for the GPU box; Ubuntu 22.04 for the CPU box. Region: whichever approves quota first (ap-south-1 or us-east-1). Stop instances when idle.

## Setup (both machines)
```bash
git clone https://github.com/Paradox-73/Business_Entity_Resolution.git && cd Business_Entity_Resolution
pip install polars pyarrow rapidfuzz scikit-learn sparse_dot_topn lightgbm xgboost==2.0.3 "transformers>=4.44" sentencepiece protobuf
mkdir -p work/models/full work/models/full_cons student_resource/dataset/train
cd code/business_entity_resolution/src          # all commands below run from here; paths default to ../../../work
```
Upload from the laptop (`aws s3 cp` via a bucket, or `scp -i key.pem`), same relative paths as in the repo:
| Files | Size | Needed by |
|---|---|---|
| `work/{train,test}_s{1,2,3}.parquet`, `work/testfr_s{1,2,3}.parquet`, `work/maps.json` | 2.2 GB | all jobs |
| `student_resource/dataset/train/train_ground_truth.tsv` | ~0.3 GB | all jobs |
| `work/models/full_cons/{oof.parquet,result.json}`, `work/test_scores_full_cons.parquet`, `work/test_scores_full_cons_testfr.parquet` | ~0.3 GB | G1 |
| `work/models/full/stage1_base.parquet` | 0.75 GB | G2 (hard negatives) |

Both scripts were smoke-tested end to end on the laptop GPU (tiny subsets, 25 Sep). On AWS run each once with a size limit first (`BER_CE_LIMIT=2000` for rerank.py, `BER_EMB_LIMIT=3000` for embed_all.py), check it finishes, delete `work/ce` / `work/emb_all`, then run fully without the limit.

## Overnight on the laptop (25→26 Sep, started ~22:30)
G1 and G2 are running now on the laptop GPU with the **small** model (multilingual-e5-small, word table frozen; the base models do not fit 4 GB for training): `work/run_overnight.sh`, log `work/run_overnight.log`. Expected finish ~06:00.
- If the small reranker gains on held-out, AWS re-runs G1 with mDeBERTa-v3-base (bigger model, same commands) to get more.
- If the small retriever lifts recall, AWS runs C1 (the rebuild) with its output; re-running G2 with e5-base is optional.
- If a small model gains nothing, its AWS re-run is skipped.

## Jobs, in order
### G1 — Reranker (GPU, start first, ~2.5 h)
A transformer (mDeBERTa-v3-base) reads the two records' text together and rescores only "close calls" (top-2 candidates of records whose best probability is 0.01–0.995, or whose 2nd candidate has stage-1 probability ≥ 0.2): 1.6M train records, 2.9M test records (3.4M pairs). A small XGBoost (stage 3) combines it with the GBDT probabilities.
```bash
python rerank.py select full_cons     # minutes
python rerank.py train                # ~1 h (bf16 on A10G)
python rerank.py score                # ~30-40 min
python rerank.py stage3               # prints held-out macro F0.5 per threshold: GBDT vs +transformer
```
Send back: the `stage3` log lines and `work/ce/test_scores_ce.parquet`.
**Gate**: keep if held-out gain ≥ +0.001 AND the laptop's tl2 check agrees (Laptop L2).
Note: `test_scores_full_cons.parquet` has France rows from the old cleaning; the laptop picks the France rows according to v7's result when it writes the submission.

### G2 — Retriever for every record (GPU, after G1's `train` step, or on a second GPU box, ~4 h)
Fine-tunes multilingual-e5-base on all train matches (+ one hard negative each), encodes all 24M records, finds the top 20 S1 rows per record.
```bash
python embed_all.py train      # ~1.5-2 h
python embed_all.py encode     # ~1-1.5 h
python embed_all.py search 20  # ~30 min
python embed_all.py eval       # held-out recall@10 / @20
```
Send back: the `eval` numbers and `work/emb_all/{train,test}_topk.parquet` (~2–3 GB).
**Gate**: continue to C1 only if the union of the current shortlist and the retriever's top 20 lifts the recall ceiling above 0.990 (the laptop computes this from `train_topk.parquet` in minutes).

### C1 — Rebuild with the retriever as a 5th search (CPU box, ~1–1.5 h estimated on 64 vCPU)
The laptop adds the 5th search to `candidates.py` on Saturday morning (reads `emb_all/*_topk.parquet`, adds an `emb2_cos` feature), smoke-tests it on one chunk and pushes it. Then on the CPU box: rebuild train, test and tl2 pairs with `pipeline.py build`, copy the pairs to the GPU box, retrain stage 1 + stage 2 there, predict test.
**Gate**: keep if it beats v6's base on tl2.

## Laptop on 26 Sep
- **L1**: the `tl2` build finishes overnight → score the existing fold models on it out-of-fold (v2, v3, v4, v6 models). tl2 is trusted only if it ranks them like the LB (v4 far below v2 < v3). Then tune the decision rule and test calibration on tl2.
- **L2**: check every AWS result on tl2 before it becomes an upload.
- **L3**: France: after v7's LB, keep v3's France rows or the cleaned ones.
- **L4**: write the submission files (`finalize.py`, `merge_country.py`), run `check_submission.py`, save each version (`work/save_version.sh`).

## Uploads on 26 Sep (5)
| # | When | File | Question it answers |
|---|---|---|---|
| U1 | morning | **v7** = v6 US/India + v3 France (ready: `submissions/v7/`) | LB − 0.9700 = consensus effect on US/India; v6 − v7 = France-cleaning effect |
| U2 | after G1 + L2 | best base + reranker | does the transformer's gain reach the LB? |
| U3 | after C1 (may slip to Sunday) | + retriever rebuild | does higher recall reach the LB? |
| U4–U5 | evening | decision rule / France variant chosen on tl2 | kept for the best combination |

Sunday: 5 more uploads for the final combination; the final zip uses the version with the best LB among those that also win on tl2.
