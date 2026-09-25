# Business Entity Resolution — pipeline (v2)

Python 3.12. One laptop is enough: 16 GB RAM, any 4 GB+ NVIDIA GPU (used only for the embedding model).
End-to-end runtime on an RTX 3050 laptop: ~8 h (most of it building features for 10M train + 10M test records).

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118   # GPU build of PyTorch
cd src
# expects the dataset at ../../../student_resource/dataset (override with BER_DATA=...)
python learn_maps.py                    # address short forms + local-script state names, learned from TRAIN pairs -> work/maps.json
python prep.py                          # clean all 6 source files -> work/*.parquet
python embed.py train                   # fine-tune multilingual-e5-small on non-Latin train names -> work/e5_ft_addr/
python embed.py encode                  # embeddings for S1 rows + non-Latin S2/S3 rows -> work/emb/
python pipeline.py build train full     # shortlist + features for all of TRAIN -> work/pairs/full/
python pipeline.py train full           # stage-1 + stage-2 LightGBM, out-of-fold validation, decision rule -> work/models/full/
python pipeline.py build test test      # shortlist + features for TEST -> work/pairs/test/
python pipeline.py predict full test    # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

Folders can be changed with env vars `BER_DATA`, `BER_WORK`, `BER_OUT`.
Analysis tools: `error_analysis.py <tag>` (points lost per mistake type), `blocking_exp.py <country> <n>` (shortlist recall/time at full density), `embed.py eval`.

## Files
- `common.py` — paths, TSV reading, integer id encoding, the macro F0.5 metric.
- `normalize.py` — name/address cleaning (accents, web domains, d/b/a aliases, legal forms, junk tokens, numbers).
- `learn_maps.py` — normalisation tables learned from training matches only.
- `embed.py` — fine-tuned multilingual embedding for non-Latin-script names.
- `candidates.py` — blocking (name char-3-grams, address words, name+address words, embeddings; same country) + 57 pair features.
- `pipeline.py` — build / train / predict: two-stage LightGBM, grouped 3-fold OOF validation, expected-F0.5 set selection.

## Models and licences
- `intfloat/multilingual-e5-small` — MIT, 118M parameters (fine-tuned here).
- LightGBM — MIT. No external data, APIs or lookups.
