import polars as pl, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import read_truth, id_to_int
T = r"C:/ber_scratch/frfix/twins"
F = r"C:/ber_scratch/france"
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
