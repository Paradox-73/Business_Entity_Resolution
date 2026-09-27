"""Follow-up on dlx_top.parquet (dl_matcher_exp.py output): wider threshold grid, and the team's design (transformer only on close calls)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import zlib
import numpy as np
import polars as pl
from common import WORK, read_truth, id_to_int, macro_f05_df
from pipeline import decide_expf

N_S1 = 40000
top = pl.read_parquet(os.path.join(WORK, "dlx_top.parquet")).with_columns(
    q=id_to_int("entity_id"), s=id_to_int("entity_id_s"), p_dl=1 / (1 + (-pl.col("ce")).exp()))
samp = []
for c in ("US", "India"):    # same sample as dl_matcher_exp.build
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country"]).filter(pl.col("country") == c)
    s1 = s1.with_columns(h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()), return_dtype=pl.Int64))
    ev = s1.filter(pl.col("h") % 1000 < 500)
    rate = min(1.0, N_S1 / ev.height)
    samp.append(ev.filter((pl.col("h") // 1000) % 10000 < rate * 10000).select("entity_id", "country"))
samp = pl.concat(samp)
truth = read_truth().join(samp.rename({"entity_id": "s1_id"}), on="s1_id").select(s=id_to_int("s1_id"), q=id_to_int("q_id"), country="country")
s1e = samp.select(s=id_to_int("entity_id"), country="country")

# team design: GBDT everywhere, stack score only on close calls (best GBDT p in [0.01, 0.995] or 2nd p >= 0.2)
g = top.sort("p_gbdt", descending=True).with_columns(r=pl.int_range(pl.len()).over("q"))
cc = g.group_by("q").agg(best=pl.col("p_gbdt").max(), second=pl.col("p_gbdt").filter(pl.col("r") == 1).max())
cc = cc.filter(pl.col("best").is_between(0.01, 0.995) | (pl.col("second").fill_null(0) >= 0.2)).select("q", cc=pl.lit(True))
top = top.join(cc, on="q", how="left").with_columns(pl.col("cc").fill_null(False))
top = top.with_columns(p_team=pl.when("cc").then(pl.col("p_stack")).otherwise(pl.col("p_gbdt")))
print(f"close-call records: {top.filter('cc')['q'].n_unique()} of {top['q'].n_unique()}")

def rules(b, prob, tr, se):
    res = {}
    t1 = b.sort(prob, descending=True).unique("q", keep="first")
    for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        res[f"thr {t}"] = macro_f05_df(t1.filter(pl.col(prob) >= t).select("s", "q"), tr, se)
    for fl in (0.1, 0.2, 0.3, 0.5):
        for a in (1.0, 1.5, 2.0):
            res[f"expF {fl}/{a}"] = macro_f05_df(decide_expf(b, prob, fl, a), tr, se)
    k = max(res, key=res.get)
    return res[k], k

for name, prob in (("GBDT", "p_gbdt"), ("DL alone", "p_dl"), ("STACK", "p_stack"), ("team design (DL on close calls)", "p_team")):
    out = []
    for c in ("all", "US", "India"):
        b = top if c == "all" else top.filter(pl.col("country") == c)
        tr = truth if c == "all" else truth.filter(pl.col("country") == c)
        se = s1e if c == "all" else s1e.filter(pl.col("country") == c)
        sc, k = rules(b, prob, tr.select("s", "q"), se.select("s"))
        out.append(f"{c} {sc:.5f} ({k})")
    print(f"{name:32s} " + " | ".join(out))
