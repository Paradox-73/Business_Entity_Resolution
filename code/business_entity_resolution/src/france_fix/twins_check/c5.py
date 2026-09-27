import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
top = pl.read_parquet(f"{T}/top_k3.parquet", columns=["q", "s", "p2", "p2g", "p3", "acc", "pat", "kq", "ks", "qcity"])
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyc = set(zip(s1["ks"].to_list(), s1["scity"].to_list()))
top = top.with_columns(twin=pl.Series([(a, c) in keyc and a != b for a, b, c in zip(top["kq"].to_list(), top["ks"].to_list(), top["qcity"].to_list())]))
e = pl.read_parquet(f"{F}/pairs_v7ens.parquet").select("q", "s").with_columns(e=pl.lit(True))
for v in ["v7g_num", "v7i", "v7j"]:
    p = pl.read_parquet(f"{F}/pairs_{v}.parquet").select("q", "s").with_columns(v=pl.lit(True))
    j = e.join(p, on=["q", "s"], how="full", coalesce=True).with_columns(pl.col("e").fill_null(False), pl.col("v").fill_null(False))
    rem = j.filter(pl.col("e") & ~pl.col("v")).join(top, on=["q", "s"], how="left")
    add = j.filter(~pl.col("e") & pl.col("v")).join(top, on=["q", "s"], how="left")
    print(f"== {v}: removed {rem.height}, added {add.height}")
    for nm, d in [("removed", rem), ("added", add)]:
        if d.height == 0: continue
        print(nm, "pattern mix (top 6):", d.group_by("pat").agg(n=pl.len(), twin=pl.col("twin").mean()).sort("n", descending=True).head(6).to_dicts())
        print(nm, "not-best-candidate pairs:", d["pat"].null_count())
