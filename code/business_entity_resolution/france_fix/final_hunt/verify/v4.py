import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int, read_tsv
R = os.path.dirname(WORK)
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
a = pl.read_parquet(r"C:/ber_scratch/final/same-address/add_france.parquet")
print("rows", a.height, "unique q", a["q"].n_unique(), "unique (s,q)", a.select("s", "q").unique().height, "nulls", a.select(pl.col("s").null_count(), pl.col("q").null_count()).row(0))
d = read_tsv(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
print("v10b rows", d.height)
v = (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
      .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
print("v10b pairs", v.height, "dup q in v10b", v.height - v["q"].n_unique())
print("pairs already in v10b:", a.join(v, on=["s", "q"]).height, "; records matched in v10b:", a.join(v.select("q").unique(), on="q").height)
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), s_country="country").join(a.lazy().select("s").unique(), on="s").collect()
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(q=id_to_int("entity_id").cast(pl.Int64), q_country="country", src=pl.lit(k)) for k in (2, 3)]).join(a.lazy().select("q"), on="q").collect()
print("S1 ids found", s1.height, "of", a["s"].n_unique(), "; record ids found", rec.height, "of", a.height, "(dups across s2/s3:", rec.height - rec["q"].n_unique(), ")")
x = a.select("s", "q").join(s1, on="s").join(rec, on="q")
print(x.group_by("s_country", "q_country").len())
# S1 ids in v10b output (must be test S1 rows)
print("S1 rows present in v10b file:", a.select("s").unique().join(v.select("s").unique(), on="s").height, "of", a["s"].n_unique(), "(others are empty rows:", (a["s1_nmatch"] == 0).sum(), "pairs)")
vetos = pl.concat([i64(pl.read_parquet(p).select("q", "s")) for p in (
    r"C:/ber_scratch/frfix/namechg/veto_set.parquet", os.path.join(WORK, "frfix2", "fp_veto_set.parquet"),
    os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet"))]).unique()
print("veto sets size", vetos.height, "; pairs in veto sets", a.join(vetos, on=["s", "q"]).height, "; records in veto sets (any S1)", a.join(vetos.select("q").unique(), on="q").height)
# already-added France recall pairs (v10b) overlap
fa = i64(pl.read_parquet(os.path.join(WORK, "out_v10b_fr", "france_recall_added.parquet")).select("q", "s"))
print("overlap with v10b's france_recall_added:", a.join(fa, on="q").height)
# gathik
g = (read_tsv(os.path.join(WORK, "gathik", "v8_matching_results.tsv")).with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
     .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
gj = a.select("s", "q").join(g.rename({"s": "gs"}), on="q", how="left")
print("gathik: same S1", (gj["gs"] == gj["s"]).sum(), " other S1", ((gj["gs"] != gj["s"]) & gj["gs"].is_not_null()).sum(), " unmatched", gj["gs"].is_null().sum())
# S1 rows getting >1 added record
print("S1 rows receiving 2+ adds:", a.group_by("s").len().filter(pl.col("len") > 1).height)
