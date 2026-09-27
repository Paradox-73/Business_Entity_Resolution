import sys, os
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, polars as pl
from common import WORK, id_to_int
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select("entity_id", fr=pl.col("country") == "France")
def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
a = rd(r"E:/Projects/Amazon ML Challenge/submissions/v7ens/matching_results.tsv")
b = rd(r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal/final/matching_results.tsv")
print("rows", a.height, b.height)
j = a.join(b, on="source1_entity_id", how="full", suffix="_b").join(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
print("missing in b", j["source1_entity_id_b"].null_count(), "missing in a", j["source1_entity_id"].null_count())
def norm(c): return pl.col(c).str.split(",").list.sort().list.join(",")
j = j.with_columns(diff=norm("matched_entity_ids") != norm("matched_entity_ids_b"))
print("diff rows France", j.filter(pl.col("fr"))["diff"].sum(), "non-France", j.filter(~pl.col("fr"))["diff"].sum())
def pairs(d, col):
    return d.filter(pl.col("fr")).select("source1_entity_id", col).with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(s=id_to_int("source1_entity_id"), q=id_to_int(col))
pa = pairs(j, "matched_entity_ids"); pb = pairs(j, "matched_entity_ids_b")
print("France pairs v7ens", pa.height, "proposal", pb.height)
print("removed", pa.join(pb, on=["s", "q"], how="anti").height, "added", pb.join(pa, on=["s", "q"], how="anti").height)
# compare to consensus mask
b2 = pl.read_parquet("C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal/pairs.parquet", columns=["q", "s", "in_ens"])
cm = np.load("C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal/mask_consensus.npy")
cons = b2.filter(pl.Series(cm)).select(pl.col("s"), pl.col("q"))
print("consensus", cons.height, "tsv-not-cons", pb.join(cons, on=["s","q"], how="anti").height, "cons-not-tsv", cons.join(pb, on=["s","q"], how="anti").height)
print("ens pairs.parquet in_ens", b2["in_ens"].sum(), "tsv v7ens vs in_ens anti", pa.join(b2.filter("in_ens"), on=["s","q"], how="anti").height)
# records matched to >1 S1?
print("dup q in proposal", pb.height - pb["q"].n_unique())
