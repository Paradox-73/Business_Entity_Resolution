# AWS plan — what runs where (25 Sep night → 27 Sep)

## Rules check (Amazon forum answers, 25 Sep)
- Allowed: MIT/Apache-2.0 open models ≤ 8B each, offline, fine-tuned only on provided data; pure-algorithm libraries (RapidFuzz, scikit-learn, LightGBM, XGBoost, pandas); small hand-written normalisation dictionaries; unsupervised statistics on the test files (TF-IDF, blocking index); self-training and synthetic pairs from provided records.
- Not allowed: external data, APIs, geocoders, libpostal, postal/gazetteer datasets, hosted LLM APIs.
- Our pipeline complies: multilingual-e5 (MIT) fine-tuned on train pairs; maps learned from train pairs; no external data.
- Newly allowed and worth using: a **hand-written French normalisation list** (rue/r, avenue/av, boulevard/bd, SARL, SAS, …), **self-training on test France**.

## Split

| Where | Job | Why there | Output |
|---|---|---|---|
| **AWS GPU** | C. Retriever for all records: fine-tune `intfloat/multilingual-e5-base` (MIT, 278M) on all train matches + hard negatives; encode 24M train+test records; top-20 S1 per record | needs 24 GB GPU + ~6 h on the laptop | `work/emb_all/{train,test}_topk.parquet` (q, s, emb2_cos) |
| **AWS GPU** | D. Cross-encoder reranker on close calls (after C) | too slow on 4 GB | score per (q, s) for top-2 close calls |
| **Laptop GPU/CPU** | A. v4 = training on test-like conditions (running) | uses cached features | v4 submission |
| **Laptop** | E. calibration of p2, per-country decision; check 4 (are test look-alikes copies of train businesses?) | cheap | v5/v6 |
| **Laptop** | B. France: French normalisation list + self-training (if v5 says France is weak) | CPU | France fix |
| **Laptop** | merge C's candidates/features into the build, rebuild, retrain | CPU-bound (string features) | v6+ |

## AWS setup (tonight)
1. **GPU quota first** (new accounts usually have 0 for GPU instances): Service Quotas → EC2 → "Running On-Demand G and VT instances" → request **8 vCPUs** (region ap-south-1 Mumbai or us-east-1). Approval can take hours — request immediately.
2. Instance: **g5.2xlarge** (1× A10G 24 GB, 8 vCPU, 32 GB RAM, ~$1.2/h). AMI: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu). Disk: 150 GB gp3.
   - Fallback if quota is slow: SageMaker Studio notebook `ml.g5.2xlarge` (has its own quota).
3. On the instance:
   ```bash
   git clone https://github.com/Paradox-73/Business_Entity_Resolution.git && cd Business_Entity_Resolution
   pip install polars pyarrow transformers==5.* rapidfuzz scikit-learn
   mkdir -p work
   ```
4. Upload from the laptop (only the cleaned files, ~3 GB): `work/train_s{1,2,3}.parquet`, `work/test_s{1,2,3}.parquet`, `work/maps.json`, `work/models/full/stage1_base.parquet` (hard negatives), and `student_resource/dataset/train/train_ground_truth.tsv`.
   - Easiest: `aws s3 cp` to a bucket, then download on the instance; or `scp -i key.pem`.
5. Run (from `code/business_entity_resolution/src`, set `BER_WORK=~/Business_Entity_Resolution/work BER_DATA=...`):
   ```bash
   python embed_all.py train      # ~1.5-2 h on A10G (2M pairs, e5-base)
   python embed_all.py encode     # ~1-1.5 h (24M records)
   python embed_all.py search 20  # ~30 min
   python embed_all.py eval       # held-out recall@10 / @20 -> tell Claude the numbers
   ```
6. Send back `work/emb_all/train_topk.parquet` and `test_topk.parquet` (~2-3 GB total). Stop the instance when idle.

## Decision points
- If `eval` recall@20 of the retriever alone > 0.985 (current shortlist ceiling), it goes into the shortlist as a 5th search + `emb2_cos` feature on every pair → rebuild.
- D starts only after C's numbers are in.
