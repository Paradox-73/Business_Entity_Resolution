# US/India stacker (v10d)

A LightGBM model that re-scores every US/India (S1 row, record) pair from per-pair features of both pipelines, then the
expected-F0.5 rule picks the matches. It replaced the fixed 0.6 / 0.4 blend of v10a.

- Features: our three cross-encoder family probabilities (e5-small, bge-reranker-v2-m3, e5-large), our blend, the second
  pipeline's probability, both pipelines' GBDT p1/p2, which pipeline scored the pair, the pair's rank and margin within its
  record, country, record without address, signed and absolute house-number difference, exact name equality. Features that
  depend on the number of candidates per S1 row were left out because test candidate lists are wider than held-out ones.
- Validation: the evaluation half of train S1 rows, split in two by S1 id; models fitted on one part score the other, plus
  models fitted on the training half only; the final score is the mean of the 4 models. Held-out macro F0.5 +0.000209 over
  the v10a blend (halves +0.000216 / +0.000202, US +0.000226, India +0.000184).
- Order: `build_ho.py` (held-out features) -> `fit.py` -> `score_avg.py` -> `build_test.py` -> `apply_test.py`
  (list-mover overrides applied after the model, then decide_expf(0.5, 1.0)). Scripts use `C:/ber_scratch` as their
  scratch folder.
