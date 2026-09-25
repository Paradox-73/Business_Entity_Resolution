"""Calibrate stage-2 probabilities (isotonic, cross-fitted by fold) and re-tune the decision per country.

  python calib.py <model_dir_name> <tag>       e.g. calib.py tlike_xgb_sib tlike

The v3 check showed p2 is under-confident in the middle (0.45 predicted -> 0.56 actual), which makes the
expected-F0.5 rule drop records it should keep. Isotonic regression maps p2 to the observed match rate.
Fits on the best candidate of each S2/S3 row (the only row a decision uses) from the OTHER folds and
applies to this fold, so the score is out-of-fold. Writes models/<model_dir>/calib.json:
  {"x": [...], "y": [...], "decision": {country: rule}, "scores": {...}}
"""
import json
import os
import sys
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from common import WORK, log, macro_f05_df
from pipeline import train_meta, decide_expf, N_FOLDS

md, tag = os.path.join(WORK, "models", sys.argv[1]), sys.argv[2]
s1, t, _ = train_meta(tag)
s1e = s1.filter("is_eval").select("s", "country")
B = pl.read_parquet(os.path.join(md, "oof.parquet"))
best = B.sort("p2", descending=True).unique("q", keep="first")
pc = np.zeros(B.height, np.float32)
fold = B["fold"].to_numpy()
p2 = B["p2"].to_numpy()
for f in range(N_FOLDS):
    tr = best.filter(pl.col("fold") != f)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(tr["p2"].to_numpy(), tr["label"].to_numpy())
    pc[fold == f] = iso.predict(p2[fold == f])
B = B.with_columns(pc=pl.Series(pc))
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(best["p2"].to_numpy(), best["label"].to_numpy())
for lo, hi in ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.8, 0.9)):
    x = best.filter((pl.col("p2") >= lo) & (pl.col("p2") < hi))
    log(f"p2 in [{lo},{hi}): n={x.height}, actual match rate {x['label'].mean():.3f}")


def thr_rule(b, prob, t_):
    return b.sort(prob, descending=True).unique("q", keep="first").filter(pl.col(prob) >= t_).select("s", "q")


res, dec, best_sc = {}, {}, {}
for c in s1e["country"].unique().sort().to_list():
    se = s1e.filter(pl.col("country") == c).select("s")
    tc = t.join(se, on="s")
    bc = B.join(s1.filter(pl.col("country") == c).select("s"), on="s")   # all S1 of c: a row's best S1 may be non-eval
    sc = {}
    for th in (0.45, 0.5, 0.55, 0.6, 0.65):
        sc[("p2", "thr", th)] = macro_f05_df(thr_rule(bc, "p2", th), tc, se)
    for prob in ("p2", "pc"):
        for floor in (0.2, 0.3, 0.4, 0.5):
            for alpha in (1.0, 1.5, 2.0):
                sc[(prob, "expf", floor, alpha)] = macro_f05_df(decide_expf(bc, prob, floor, alpha), tc, se)
    for th in (0.45, 0.5, 0.55, 0.6):
        sc[("pc", "thr", th)] = macro_f05_df(thr_rule(bc, "pc", th), tc, se)
    top = sorted(sc.items(), key=lambda kv: -kv[1])
    log(f"{c}: eval S1 {se.height}; best 6: " + "; ".join(f"{k} {v:.5f}" for k, v in top[:6]))
    log(f"{c}: current rule p2 thr 0.55 {sc[('p2', 'thr', 0.55)]:.5f}")
    res[c] = {" ".join(map(str, k)): v for k, v in sc.items()}
    k, best_sc[c] = top[0]
    dec[c] = ({"type": "thr", "t": k[2], "prob": k[0]} if k[1] == "thr" else
              {"type": "expf", "floor": k[2], "alpha": k[3], "prob": k[0]})
n = {c: s1e.filter(pl.col("country") == c).height for c in res}
# 'default' (countries without labels, i.e. France): the single rule that is best over all eval S1 pooled
pooled = {k: sum(res[c][k] * n[c] for c in res) / sum(n.values()) for k in res[next(iter(res))]}
kd = max(pooled, key=pooled.get).split()
dec["default"] = ({"type": "thr", "t": float(kd[2]), "prob": kd[0]} if kd[1] == "thr" else
                  {"type": "expf", "floor": float(kd[2]), "alpha": float(kd[3]), "prob": kd[0]})
log(f"pooled best rule (used for France): {' '.join(kd)} {pooled[' '.join(kd)]:.5f}")
tot = sum(best_sc[c] * n[c] for c in res) / sum(n.values())
base = sum(res[c]["p2 thr 0.55"] * n[c] for c in res) / sum(n.values())
log(f"ALL eval S1: current rule (p2 thr 0.55) {base:.5f} -> per-country best {tot:.5f}  (+{tot - base:.5f}); {dec}")
json.dump({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(), "decision": dec,
           "base": base, "best": tot, "scores": res}, open(os.path.join(md, "calib.json"), "w"), indent=1)
