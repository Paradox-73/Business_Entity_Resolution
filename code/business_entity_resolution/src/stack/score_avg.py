"""Held-out check of the 4-model mean: macro F0.5 of p (the v10a blend), pst (the _rob models), pst2 (the _robtr
models) and their mean pa, on each held-out subset -> OUT/res_avg.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *
from common import macro_f05_df, log
from pipeline import decide_expf
a = pl.read_parquet(os.path.join(OUT, "ho_pst_rob.parquet")); b = pl.read_parquet(os.path.join(OUT, "ho_pst_robtr.parquet"), columns=["q", "s", "pst"]).rename({"pst": "pst2"})
d = a.join(b, on=["q", "s"]).with_columns(pa=((pl.col("pst") + pl.col("pst2")) / 2).cast(pl.Float32))
ev = pl.read_parquet(os.path.join(OUT, "ev.parquet")); tr = pl.read_parquet(os.path.join(OUT, "truth_ev.parquet"))
SUB = {"all": ev.select("s")}
for h in (0, 1):
    for c, cn in ((0, "US"), (1, "IN")):
        SUB[f"h{h}{cn}"] = ev.filter((pl.col("half") == h) & (pl.col("ctry") == c)).select("s")
for h in (0, 1):
    SUB[f"h{h}"] = ev.filter(pl.col("half") == h).select("s")
for c, cn in ((0, "US"), (1, "IN")):
    SUB[cn] = ev.filter(pl.col("ctry") == c).select("s")
res = {}
for col in ("p", "pst", "pst2", "pa"):
    pr = decide_expf(d.select("q", "s", p2=col), "p2", 0.5, 1.0)
    res[col] = {k: macro_f05_df(pr, tr, v) for k, v in SUB.items()}
    log(f"{col:5s} " + " ".join(f"{k} {1e5 * (v - res['p'][k]):+.1f}" for k, v in res[col].items()))
import json; json.dump(res, open(os.path.join(OUT, "res_avg.json"), "w"), indent=1)
