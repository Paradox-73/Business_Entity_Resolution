"""Methodology section 4 ("Rules used"): sizes and saved columns of the France pair sets in sets/.

  python set_summary.py        (from any folder; prints; BER_SETS overrides the sets folder)

Prints, per set file, its rows and columns; for recall_add_set.parquet the estimated true share `est` by group and its
mean (0.93); for fp_veto_set.parquet the pairs, the mean estimated true share `t` and the all-lowercase share `low` of
each tier (A 846 / B 232 / C 67; tier A lowercase 3.4%); and for desc_veto_set.parquet the pairs by kind and by house
number.
"""
import os

import polars as pl

SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sets"))
pl.Config.set_tbl_rows(40)
for f in sorted(os.listdir(SETS)):
    if f.endswith(".parquet"):
        d = pl.read_parquet(os.path.join(SETS, f))
        print(f, d.height, d.columns)
r = pl.read_parquet(os.path.join(SETS, "recall_add_set.parquet"))
print(r.group_by("grp").agg(n=pl.len(), est=pl.col("est").mean()).sort("n", descending=True))
print("recall_add_set: mean estimated true share", round(r["est"].mean(), 4))
fp = pl.read_parquet(os.path.join(SETS, "fp_veto_set.parquet"))
print(fp.group_by("tier").agg(n=pl.len(), t=pl.col("t").mean(), low=pl.col("low").cast(pl.Float64).mean()).sort("tier"))
d = pl.read_parquet(os.path.join(SETS, "desc_veto_set.parquet"))
print(d.group_by("kind").len().sort("len", descending=True))
print(d.group_by("num").len().sort("len", descending=True))
