"""Swap matrix: dropped S1 word -> added record word; lift vs independence; number keeping; acceptance/restored."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
OUT = f"{SCRATCH}/frfix/namechg"
c = pl.read_parquet(f"{OUT}/fr_chg.parquet").drop("qn", "qa", "sa", "sn_2")
c = c.with_columns(num=pl.col("pat").str.split("|").list.last(), nk=pl.col("pat").str.split("|").list.first())
sw = c.filter((pl.col("nk") == "swap") & (pl.col("added").list.len() == 1) & (pl.col("dropped").list.len() == 1)).with_columns(
    a=pl.col("added").list.first(), d=pl.col("dropped").list.first())
print("single-word swaps:", sw.height, "of swap rows", c.filter(pl.col("nk") == "swap").height)
sn = sw.filter(pl.col("num") == "nsame")
N = sn.height
ca = sn.group_by("a").agg(na=pl.len()); cd = sn.group_by("d").agg(nd=pl.len())
m = sn.group_by("d", "a").agg(n=pl.len(), acc=pl.col("acc").mean(), rest=(pl.col("in7m") & ~pl.col("acc")).mean(),
                               p3=pl.col("p3").mean(), g=pl.col("p2g").mean())
m = m.join(ca, on="a").join(cd, on="d").with_columns(lift=pl.col("n") * N / (pl.col("na") * pl.col("nd")))
# number keeping for the same (d,a) swap
k = sw.filter(pl.col("num") != "nmiss").group_by("d", "a").agg(keep=(pl.col("num") == "nsame").mean(), nk=pl.len())
m = m.join(k, on=["d", "a"], how="left")
print("top (d,a) swaps at same number:")
print(m.sort("n", descending=True).head(60))
print("added words at same number (swap):")
print(ca.sort("na", descending=True).head(40).to_dicts())
m.write_parquet(f"{OUT}/swap_matrix.parquet")
