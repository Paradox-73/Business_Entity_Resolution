"""Build France best-candidate analysis table: street match, added/dropped words, twin/sibling flags, v7m/v7ens membership."""
import sys, re, collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from rapidfuzz import fuzz
import france_cal as fc
from fr_restore import street
from common import WORK, id_to_int

FR = f"{SCRATCH}/france"
OUT = f"{SCRATCH}/frfix/namechg"

f = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "p2", "p2g", "qn", "qa", "sn", "sa", "pat", "acc", "s_2", "p2_2", "sn_2"])
print("fr_top", f.height)
# p3 (stage-3 blend) for the best candidate
p3 = (pl.scan_parquet(f"{WORK}/test_scores_blend_ab_a2.parquet").select("q", "s", pl.col("p2").alias("p3"))
      .join(f.select("q", "s").lazy(), on=["q", "s"], how="semi").collect())
f = f.join(p3, on=["q", "s"], how="left")
print("p3 attached", f["p3"].null_count())

# v7m France pairs
sub = pl.read_csv(f"{ROOT}/submissions/v7m/matching_results.tsv", separator="\t", quote_char=None,
                  infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
frs = f.select("s").unique()
v7m = (sub.with_columns(s=id_to_int("source1_entity_id")).join(frs, on="s")
       .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select("s", q=id_to_int("matched_entity_ids")))
print("v7m France pairs", v7m.height)
v7m.write_parquet(f"{OUT}/pairs_v7m.parquet")
v7e = pl.read_parquet(f"{FR}/pairs_v7ens.parquet")
f = (f.join(v7m.with_columns(in7m=pl.lit(True)), on=["q", "s"], how="left")
      .with_columns(pl.col("in7m").fill_null(False)))
print("v7m pairs that are best candidates:", f["in7m"].sum(), "acc (v7ens):", f["acc"].sum())

# name diffs for all rows (fast enough? 1.43M rapidfuzz loops) -> restrict to name-change patterns with nsame/nmiss plus sample
keep = f.filter(~pl.col("pat").str.starts_with("same|") & ~pl.col("pat").str.starts_with("squashed|")
                & ~pl.col("pat").str.starts_with("acronym|"))
print("name-change rows", keep.height)


def diff(sn, qn):
    ts, tq = fc.toks(sn), fc.toks(qn)
    dropped = [w for w in ts if not any(fuzz.ratio(w, b) >= 80 for b in tq)]
    added = [w for w in tq if not any(fuzz.ratio(w, x) >= 80 for x in ts)]
    return added, dropped


ad, dr, sm = [], [], []
for sn, qn, sa, qa in zip(keep["sn"].to_list(), keep["qn"].to_list(), keep["sa"].to_list(), keep["qa"].to_list()):
    a, d = diff(sn, qn)
    ad.append(a); dr.append(d)
    x, y = street(sa), street(qa)
    sm.append(bool(x[0] is not None and x[0] == y[0] and x[1] and y[1] and fuzz.token_sort_ratio(x[1], y[1]) >= 85))
keep = keep.with_columns(added=pl.Series(ad, dtype=pl.List(pl.Utf8)), dropped=pl.Series(dr, dtype=pl.List(pl.Utf8)),
                         samestreet=pl.Series(sm))

# normalized name keys: token set of the full name (legal forms kept as tokens? toks() drops them) -> sorted join
def key(x):
    return " ".join(sorted(set(fc.toks(x))))


s1 = pl.scan_parquet(f"{WORK}/test_s1.parquet").filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id"), n="business_name", a="business_address").collect()
s1k = s1.with_columns(k=pl.Series([key(x) for x in s1["n"].to_list()]))
kc = s1k.group_by("k").agg(ns1=pl.len(), s1ids=pl.col("s"))
keep = keep.with_columns(qk=pl.Series([key(x) for x in keep["qn"].to_list()]))
keep = keep.join(kc.rename({"k": "qk"}), on="qk", how="left").with_columns(
    twin=pl.col("ns1").fill_null(0) - pl.col("s1ids").list.contains(pl.col("s")).fill_null(False).cast(pl.Int64))
# twin at same number as the record?
s1num = dict(zip(s1["s"].to_list(), [street(a)[0] for a in s1["a"].to_list()]))
s1k.select("s", "k").write_parquet(f"{OUT}/s1keys.parquet")
tw_same = []
for ids, qa, s in zip(keep["s1ids"].to_list(), keep["qa"].to_list(), keep["s"].to_list()):
    if not ids:
        tw_same.append(False); continue
    n = street(qa)[0]
    tw_same.append(any(i != s and s1num.get(i) == n and n is not None for i in ids))
keep = keep.with_columns(twin_samenum=pl.Series(tw_same)).drop("s1ids")
keep.write_parquet(f"{OUT}/fr_chg.parquet")
# S1 token document frequency
df = collections.Counter()
for x in s1["n"].to_list():
    df.update(set(fc.toks(x)))
pl.DataFrame({"w": list(df.keys()), "df_s1": list(df.values())}).write_parquet(f"{OUT}/s1_df.parquet")
f.select("q", "s", "p2", "p2g", "p3", "pat", "acc", "in7m").write_parquet(f"{OUT}/fr_all.parquet")
print("done")
