import polars as pl, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import read_truth, id_to_int
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1.parquet")
k = s1.select("s").join(acc.group_by("s").len(), on="s", how="left").fill_null(0)
print(k["len"].value_counts().sort("len").with_columns(share=pl.col("count")/k.height))
tr = read_truth()
print(tr.head(3))
t = tr.group_by("s1_id").agg(n=pl.col("q_id").drop_nulls().len())
print(t["n"].value_counts().sort("n").with_columns(share=pl.col("count")/t.height))
print(s1.sample(25, seed=1))
