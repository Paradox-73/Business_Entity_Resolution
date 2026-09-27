"""Lowercase-share true-rate estimates for candidate ADD groups (rejected in v7ens) by class/kind/g band/in7m."""
import polars as pl, numpy as np
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
RD = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/refute_desc"
NC = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"
a = pl.read_parquet(f"{RD}/fr_low.parquet")
ss = pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "samestreet", "p2"])
a = a.join(ss, on=["q", "s"], how="left")
print(a.height, a.columns)
Ld, Lt = 0.0338, 0.0008
def est(x):
    return x.agg(n=pl.len(), low=pl.col("low").mean(), p3=pl.col("p3").mean(), g=pl.col("p2g").mean(), p2=pl.col("p2").mean()).with_columns(
        t=((Ld - pl.col("low")) / (Ld - Lt)).round(2),
        se=((pl.col("low").clip(1e-4) * (1 - pl.col("low")) / pl.col("n")).sqrt() / (Ld - Lt)).round(2))
chg = a.filter(pl.col("kind").is_in(["swap", "added", "other", "dropped"]))
R = chg.filter(~pl.col("acc"))
print("--- rejected (not in v7ens), by cls/kind/num")
print(est(R.group_by("cls", "kind", "num")).filter(pl.col("n") >= 300).sort("n", descending=True))
print("--- rejected N/G nsame by in7m, samestreet, g band")
x = R.filter(pl.col("cls").is_in(["N", "G"]) & (pl.col("num") == "nsame"))
print(est(x.group_by("cls", "kind", "in7m", "samestreet")).sort("cls", "kind", "in7m", "samestreet"))
print(est(x.group_by("cls", pl.col("p2g").cut([0.1, 0.3, 0.5, 0.8, 0.95]).alias("gb"))).sort("cls", "gb"))
print(est(x.group_by("cls", pl.col("p3").cut([0.01, 0.1, 0.3, 0.5]).alias("pb"))).sort("cls", "pb"))
print("--- accepted by cls/kind/num (for comparison)")
A = chg.filter(pl.col("acc"))
print(est(A.group_by("cls", "kind", "num")).filter(pl.col("n") >= 500).sort("n", descending=True))
