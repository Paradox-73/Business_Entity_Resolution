import polars as pl, os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import WORK
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(320)
u = pl.read_parquet(f"{T}/us_top_k.parquet")
e = u.filter(pl.col("nx_city") > 0)
print(e.group_by("pat").agg(n=pl.len(), rate=pl.col("label").mean(), p=pl.col("p").mean(), acc=(pl.col("p") >= 0.5).mean(), p2g=pl.col("p2g").mean(), p3=pl.col("p3").mean(),
      true_to_x=pl.col("true_in_x").mean(), none=pl.col("ts").is_null().mean()).sort("n", descending=True).head(15))
print(pl.read_parquet_schema(os.path.join(WORK, "models", "full_cons_ce", "oof.parquet")))
oof = pl.scan_parquet(os.path.join(WORK, "models", "full_cons_ce", "oof.parquet")).select("q", "s").join(e.select("q").lazy(), on="q").collect()
print("US ex records: candidates per record", oof.group_by("q").len()["len"].value_counts().sort("len"))
