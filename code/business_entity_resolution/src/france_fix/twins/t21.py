import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
F = rf"{SCRATCH}/france"
T = rf"{SCRATCH}/frfix/twins"
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(330)
u = pl.read_parquet(f"{F}/usi_top.parquet", columns=["q", "s", "label", "p", "c", "pat"])
u = u.with_columns(kc=((pl.col("p") >= 0.9).sum().over("s") - (pl.col("p") >= 0.9).cast(pl.Int64)),
                   pb=pl.col("p").cut([0.1, 0.3, 0.5, 0.7, 0.9, 0.99]))
r = u.filter(pl.col("p") >= 0.1).group_by("pb", pl.col("kc").clip(0, 7).alias("kc")).agg(n=pl.len(), p=pl.col("p").mean(), rate=pl.col("label").mean()).sort("pb", "kc")
print(r.with_columns(gap=pl.col("rate") - pl.col("p")))
# France: same count structure
top = pl.read_parquet(f"{T}/top_k3.parquet", columns=["q", "s", "p2", "acc", "pat"])
top = top.with_columns(kc=((pl.col("p2") >= 0.9).sum().over("s") - (pl.col("p2") >= 0.9).cast(pl.Int64)))
print(top.filter(pl.col("p2").is_between(0.5, 0.9)).group_by(pl.col("kc").clip(0, 7)).agg(n=pl.len(), acc=pl.col("acc").mean(), p=pl.col("p2").mean()).sort("kc"))
u2 = u.filter(pl.col("p").is_between(0.5, 0.9))
print("US/India p in [0.5,0.9): by kc", u2.group_by(pl.col("kc").clip(0, 7)).agg(n=pl.len(), p=pl.col("p").mean(), rate=pl.col("label").mean()).sort("kc"))
