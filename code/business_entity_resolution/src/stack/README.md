# US/India stacker (v10d)

A LightGBM model that re-scores every US/India (S1 row, record) pair from per-pair features of both pipelines, then the
expected-F0.5 rule picks the matches. It replaced the fixed 0.6 / 0.4 blend of v10a.

- Features: our three cross-encoder family probabilities (e5-small, bge-reranker-v2-m3, e5-large), our blend, the second
  pipeline's probability, both pipelines' GBDT p1/p2, which pipeline scored the pair, the pair's rank and margin within its
  record, country, record without address, signed and absolute house-number difference, exact name equality. Three
  list-length features are left out: the record's candidate count (`nq`) and the pair's rank and margin within the S1
  row (`rank_s`, `margin_s`). Test has more candidates per S1 row than held-out (7.07 vs 5.35-5.68).
- Validation: the evaluation half of train S1 rows, split in two by S1 id; models fitted on one part score the other, plus
  models fitted on the training half only; the final score is the mean of the 4 models. Held-out macro F0.5 +0.000209 over
  the v10a blend (halves +0.000216 / +0.000202, US +0.000226, India +0.000184).
- Order: `build_ho.py` (held-out features) -> `fit.py` (twice) -> `score_avg.py` -> `merge_models.py` -> `build_test.py`
  -> `apply_test.py`
  (list-mover overrides applied after the model, then decide_expf(0.5, 1.0)).
- Folders (`feats.py`), each settable by an environment variable:
  - `BER_STACK_DIR`: this stacker's own files (`ho.parquet`, `te.parquet`, models, results); default `$BER_WORK/stack`,
    created when missing.
  - `BER_BLEND`: output of `../blend_second.py`; default `$BER_WORK/blend9`.
  - `BER_MOVERS`: `tier12_removed.parquet` / `tier12_restored.parquet` of `../movers/build_movers.py`; default
    `$BER_WORK/movers` (copies ship in `../../sets/`).
  - `BER_STACK_MODELS` (`apply_test.py` only): the model file; default `$BER_STACK_DIR/lgb_models<tag>.pkl`. The v10d
    file ships as `../../sets/lgb_models_avg.pkl`.
- `../build_final.py` runs `build_test.py` and `apply_test.py _avg` with these variables set; the fitting steps
  (`build_ho.py`, `fit.py`, `score_avg.py`, `merge_models.py`) are not rerun there.

Commands of the v10d run (from this folder):

```bash
python build_ho.py                                             # OUT/ho.parquet, ev.parquet, truth_ev.parquet
DROP=nq,rank_s,margin_s TAG=_rob python fit.py gbdt            # 2 models, one per part of the evaluation half
DROP=nq,rank_s,margin_s TAG=_robtr FITON=1 python fit.py gbdt  # 2 models fitted on the training half
python score_avg.py                                            # held-out check of the 4-model mean -> OUT/res_avg.json
python merge_models.py _rob _robtr _avg                        # OUT/lgb_models_avg.pkl (= ../../sets/lgb_models_avg.pkl)
python build_test.py                                           # OUT/te.parquet
python apply_test.py _avg                                      # OUT/usi_pairs_avg.parquet, added/removed_vs_v10c_avg.parquet
```

`added_vs_v10c_avg.parquet` and `removed_vs_v10c_avg.parquet` hold the same (s, q) pairs as
`../../sets/usi_stack_add.parquet` (7,021) and `../../sets/usi_stack_remove.parquet` (1,275).

The output of the two `fit.py` runs of 27 Sep 2026 is in `../logs/stack_fit_rob.log` and `stack_fit_robtr.log` (each
model 20-42 s); their held-out results and that of `score_avg.py` are `../logs/stack_res_*.json`. The commands above were
reconstructed from the saved models' feature lists and file tags.
