import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1.parquet")
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet").with_columns(a=pl.lit(True))
print("cands per record:", b.group_by("q").len()["len"].value_counts().sort("len").head(10))
print("cands per S1:", b.group_by("s").len()["len"].describe())
print("S1 with no cands:", s1.height - b["s"].n_unique())
# accepted per S1
k = s1.select("s").join(acc.group_by("s").len(), on="s", how="left").fill_null(0)
vc = k["len"].value_counts().sort("len").with_columns(share=pl.col("count")/k.height)
print(vc.head(15))
print("mean accepted per S1", k["len"].mean())
