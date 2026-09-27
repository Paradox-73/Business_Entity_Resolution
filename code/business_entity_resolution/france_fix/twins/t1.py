import os, sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import WORK, id_to_int
OUT = r"C:/ber_scratch/frfix/twins"
s1 = (pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(pl.col("country") == "France")
      .select(s=id_to_int("entity_id"), sn="business_name", sa="business_address").collect())
print("S1", s1.height)
s1.write_parquet(f"{OUT}/fr_s1.parquet")
S = s1.select("s").lazy()
def sc(path, name):
    return pl.scan_parquet(path).select("q", "s", pl.col("p2").alias(name)).join(S, on="s")
g = sc(os.path.join(WORK, "test_scores_full_cons.parquet"), "g")
p3 = sc(os.path.join(WORK, "test_scores_blend_ab_a2.parquet"), "p3")
pf = sc(os.path.join(WORK, "ce/test_scores_blend_ab_a2_frmin.parquet"), "pf")
pr = sc(os.path.join(WORK, "ce/test_scores_blend_ab_a2_frrest.parquet"), "pr")
b = g.join(p3, on=["q", "s"], how="left").join(pf, on=["q", "s"], how="left").join(pr, on=["q", "s"], how="left").collect()
print(b.shape, b.null_count())
b.write_parquet(f"{OUT}/fr_pairs.parquet")
Q = b.select("q").unique()
recs = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(q=id_to_int("entity_id"), qn="business_name", qa="business_address", qc="country", src=pl.lit(k)).join(Q.lazy(), on="q").collect() for k in (2, 3)])
print("recs", recs.height, recs["qc"].value_counts())
recs.write_parquet(f"{OUT}/fr_recs.parquet")
