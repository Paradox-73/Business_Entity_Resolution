import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int, read_tsv
R = os.path.dirname(WORK)
P = r"C:/ber_scratch/final/fr-gathik-rest/fr_noise_in.parquet"
d = pl.read_parquet(P).select("s", "q")
print("pairs", d.height, "unique pairs", d.unique().height, "unique q", d["q"].n_unique(), "unique s", d["s"].n_unique(), d.schema)
i64 = lambda x: x.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
def pairs(p):
    t = read_tsv(p)
    return (t.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
print("v10b pairs", v.height, "q dup in v10b", v.height - v["q"].n_unique())
print("pairs already in v10b:", d.join(v, on=["s", "q"]).height)
print("records matched in v10b (to any S1):", d.join(v.select("q"), on="q").height)
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), sc="country").join(d.select("s").unique().lazy(), on="s").collect()
print("S1 found", s1.height, "of", d["s"].n_unique(), s1.group_by("sc").len())
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(q=id_to_int("entity_id").cast(pl.Int64), qc="country").join(d.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
print("records found", rec.height, rec.group_by("qc").len())
x = d.join(s1, on="s", how="left").join(rec, on="q", how="left")
print("country mismatch", x.filter(pl.col("sc") != pl.col("qc")).height, "null", x.filter(pl.col("sc").is_null() | pl.col("qc").is_null()).height)
vs = [r"C:/ber_scratch/frfix/namechg/veto_set.parquet", os.path.join(WORK, "frfix2", "fp_veto_set.parquet"),
      os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet")]
for p in vs:
    t = i64(pl.read_parquet(p, columns=["q", "s"]))
    print(os.path.basename(p), t.height, "pair hits", d.join(t, on=["s", "q"]).height, "record hits (any S1)", d.join(t.select("q").unique(), on="q").height)
# in our candidate lists?
lists = pl.concat([i64(pl.scan_parquet(os.path.join(WORK, f)).select("q", "s").join(d.select("q").unique().lazy(), on="q").collect()) for f in
                   ("test_scores_full_cons.parquet", "ce_x/test_rows.parquet", "test_scores_full_cons_test_b2.parquet", "frfix3/test_scores_v9y_combo.parquet")]).unique()
print("pairs in our lists:", d.join(lists, on=["s", "q"]).height, "; records having any candidate in our lists:", d.join(lists.select("q").unique(), on="q").height)
# records also in 4825 v10b additions?
add = i64(pl.read_parquet(os.path.join(WORK, "out_v10b_fr", "france_recall_added.parquet"), columns=["s", "q"]))
print("overlap with v10b recall added (q):", d.join(add.select("q"), on="q").height)
# gathik's match: is each pair in gathik's matched file?
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv"))
print("pairs in gathik matched file:", d.join(g, on=["s", "q"]).height)
