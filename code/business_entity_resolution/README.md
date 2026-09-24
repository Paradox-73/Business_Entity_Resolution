# Business Entity Resolution — pipeline

Python 3.12, CPU only (16 GB RAM laptop is enough; ~1.5 h end to end).

```bash
pip install -r requirements.txt
cd src
# expects the dataset at ../../../student_resource/dataset (override with BER_DATA=...)
python learn_maps.py        # learn address short forms + local-script state names from TRAIN pairs -> work/maps.json
python prep.py              # clean all 6 source files -> work/*.parquet
python pipeline.py dev 0.1  # validate on 10% of TRAIN: blocking recall, OOF macro F0.5, threshold; trains work/model.txt
python pipeline.py test     # run on TEST -> output/matching_results.tsv, output/candidate_pairs.tsv
```

Folders can be changed with env vars `BER_DATA`, `BER_WORK`, `BER_OUT`.

## Files
- `common.py` — paths, TSV reading, the macro F0.5 metric.
- `normalize.py` — name/address cleaning (accents, web domains, d/b/a aliases, legal forms, junk tokens, numbers).
- `learn_maps.py` — learns normalisation tables from training matches only.
- `candidates.py` — blocking (char-3-gram TF-IDF on name and address, top-10 each, same country) + pair features.
- `pipeline.py` — LightGBM pair model, out-of-fold validation, threshold tuning, test inference, output files.

## Method (v1)
1. Blocking from each S2/S3 row to S1 rows of the same country; union of name top-10 and address top-10.
2. 40 pair features: TF-IDF cosines, fuzzy name/address similarities, house-number agreement, lengths,
   flags (domain, alias, non-Latin script, missing address), chain-name count, competition among a row's candidates.
3. LightGBM binary classifier.
4. Each S2/S3 row is assigned to its single best S1 candidate if probability >= tuned threshold.
