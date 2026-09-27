"""Label-free test: number of OTHER same-name accepted records of the record's best S1, by class; calibrated on US/India labels."""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
OUT = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
# US/India
u = pl.read_parquet(f"{FR}/usi_top.parquet", columns=["q", "s", "label", "p", "pat", "country"])
u = u.with_columns(sameacc=(pl.col("pat").str.starts_with("same|") & (pl.col("p") >= 0.5)))
u = u.with_columns(oth=pl.col("sameacc").sum().over("s") - pl.col("sameacc").cast(pl.Int64))
uc = pl.read_parquet(f"{OUT}/usi_chg.parquet", columns=["q", "added"])
u = u.join(uc, on="q", how="left")
print("US/India: mean other same-name accepted records, by pattern and label")
print(u.filter(pl.col("pat").is_in(["swap|nsame", "added|nsame", "added|ndiff", "swap|ndiff", "same|ndiff", "same|nsame"]))
      .group_by("country", "pat", "label").agg(n=pl.len(), oth=pl.col("oth").mean(), oth0=(pl.col("oth") == 0).mean()).sort("country", "pat", "label"))
# France
a = pl.read_parquet(f"{OUT}/fr_all_cls.parquet")
a = a.with_columns(sameacc=((pl.col("cls") == "samename") & pl.col("acc")))
a = a.with_columns(oth=pl.col("sameacc").sum().over("s") - pl.col("sameacc").cast(pl.Int64))
pc = pl.read_parquet(f"{OUT}/fr_chg.parquet", columns=["q", "pat", "added", "dropped"])
a = a.join(pc.select("q", "dropped"), on="q", how="left").with_columns(
    kind=pl.col("pat").str.split("|").list.first())
print("France: mean other same-name accepted records by class, pattern kind, number, accepted")
print(a.filter(pl.col("cls").is_in(["D", "G", "N", "X", "samename", "drop"]) & pl.col("kind").is_in(["swap", "added", "same", "dropped"]))
      .group_by("cls", "kind", pl.col("pat").str.split("|").list.last().alias("num"), "acc")
      .agg(n=pl.len(), oth=pl.col("oth").mean(), oth0=(pl.col("oth") == 0).mean()).filter(pl.col("n") >= 300).sort("cls", "kind", "num", "acc"))
