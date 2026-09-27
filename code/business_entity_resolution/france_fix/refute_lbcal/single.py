import numpy as np, polars as pl
D = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/refute_lbcal"
bb = pl.read_parquet(f"{D}/bb.parquet")
k = bb.filter("ens").group_by("s").agg(k=pl.len())
bb = bb.join(k, on="s", how="left").with_columns(pl.col("k").fill_null(0))
for lab in ["rem", "gnR", "ens"]:
    d = bb.filter(lab)
    print(lab, d.height, "accepted pairs in its S1 (v7ens): share k=1", round((d["k"] == 1).mean(), 3), "k=2", round((d["k"] == 2).mean(), 3), "k>=3", round((d["k"] >= 3).mean(), 3), "mean k", round(d["k"].mean(), 2))
d = bb.filter(pl.col("gnR") & ~pl.col("rem"))
print("gnR kept by proposal", d.height, "share k=1", round((d["k"] == 1).mean(), 3), "mean pi", round(d["pi"].mean(), 3))
d = bb.filter(pl.col("gnR") & pl.col("rem"))
print("gnR removed by proposal", d.height, "share k=1", round((d["k"] == 1).mean(), 3), "mean pi", round(d["pi"].mean(), 3))
