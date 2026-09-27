import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int
T = r"C:/ber_scratch/final/fr-committee"
c = pl.read_parquet(os.path.join(T, "cand.parquet")).filter(pl.col("kind") == "add")
c = c.with_columns(cls=pl.when(pl.col("ops").str.contains("n_swap:desc|n_add:desc")).then(pl.lit("desc"))
                   .when(pl.col("ops").str.contains("n_swap:|n_add:|n_drop:")).then(pl.lit("word_other"))
                   .when(pl.col("ops").str.contains("n_legal_change|n_legal_add")).then(pl.lit("legal"))
                   .otherwise(pl.lit("noword")),
                   aops=pl.col("ops").str.extract_all(r"a_[a-z0-9_:]+").list.join("+"))
# twins: other France S1 rows with the same name_core
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "name_core", "addr_nums"]).filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id").cast(pl.Int64), nc="name_core", an="addr_nums")
cnt = s1.group_by("nc").agg(n_same=pl.len())
c = c.join(s1.select("s", "nc"), on="s", how="left").join(cnt, on="nc", how="left")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(55)
for k in ("noword", "word_other"):
    d = c.filter(pl.col("cls") == k)
    print(k, d.height, "twin-name share", (d["n_same"] > 1).mean())
    print(d.group_by("aops").agg(n=pl.len(), po=pl.col("po").mean(), pg=pl.col("pg").mean(), low=pl.col("low").mean(), tw=(pl.col("n_same") > 1).mean()).sort("n", descending=True).head(20))
    print(d.group_by("nm").agg(n=pl.len(), po=pl.col("po").mean(), pg=pl.col("pg").mean(), low=pl.col("low").mean(), tw=(pl.col("n_same") > 1).mean()).sort("n", descending=True).head(20))
    print(d.sample(40, seed=1).select("qn", "sn", "qa", "sa", "po", "pg", "aops"))
