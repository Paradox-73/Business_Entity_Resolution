"""Pair-level diff of each spliced file vs v9y: only France rows differ, and the changes are the expected sets."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
import polars as pl
from common import WORK, id_to_int
R = f"{ROOT}/"
W = R + "work/frfix3/"
s1c = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}).with_columns(pl.col("s1").cast(pl.Utf8))
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0).rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}).with_columns(pl.col("m").fill_null(""))
def pairs(t):
    return (t.with_columns(q=pl.col("m").str.split(",")).explode("q").filter(pl.col("q").fill_null("") != "")
             .select("s1", "q").join(s1c, on="s1").with_columns(si=id_to_int("s1"), qi=id_to_int("q")))
base = rd(R + "submissions/v9y/matching_results.tsv")
bp = pairs(base)
print("v9y pairs per country:", dict(bp.group_by("country").len().iter_rows()))
add = pl.read_parquet(W + "recall_add_set.parquet").select(si="s", qi="q", grp="grp")
veto = pl.concat([pl.read_parquet(W + "moreveto_stem_set.parquet").select(si="s", qi="q"),
                  pl.DataFrame({"si": [10659305782, 10354232030], "qi": [30565799217, 30369260852]})])
res = {}
for n in ["stem2", "recall", "combo"]:
    t = rd(W + f"out_{n}_on_v9y/matching_results.tsv")
    assert t["s1"].equals(base["s1"]), "row order differs"
    d = t.with_columns(b=base["m"]).filter(pl.col("m") != pl.col("b")).join(s1c, on="s1")
    tp = pairs(t)
    gone = bp.join(tp, on=["si", "qi"], how="anti")
    new = tp.join(bp, on=["si", "qi"], how="anti")
    res[n] = (gone, new)
    print(f"\n== {n}: rows differing by country {dict(d.group_by('country').len().iter_rows())}")
    print("  pairs per country:", dict(tp.group_by("country").len().iter_rows()))
    print(f"  France pairs {tp.filter(pl.col('country')=='France').height} (v9y {bp.filter(pl.col('country')=='France').height});"
          f" removed {gone.height}, added {new.height}")
    print(f"  removed that are veto pairs {gone.join(veto, on=['si','qi']).height}, other removed {gone.join(veto, on=['si','qi'], how='anti').height}")
    print(f"  added that are recall adds {new.join(add, on=['si','qi']).height}, other added {new.join(add, on=['si','qi'], how='anti').height}")
    oth = gone.join(veto, on=["si", "qi"], how="anti")
    if oth.height:
        print("  other removed sit on an S1 that got an add:", oth.join(new.select("s1").unique(), on="s1").height, "of", oth.height)
    # records matched to two S1 (the rule 'each record matches at most one S1')
    print("  records in two S1 lists:", tp.group_by("q").len().filter(pl.col("len") > 1).height)
# combined = union of the two alone?
gc, nc = res["combo"]; gr, nr = res["recall"]; gs, ns = res["stem2"]
gu = pl.concat([gr, gs]).unique(); nu = pl.concat([nr, ns]).unique()
print("\ncombo removed == recall removed + stem2 removed:", gc.sort(["s1","q"]).equals(gu.sort(["s1","q"])),
      "| combo added == union:", nc.sort(["s1","q"]).equals(nu.sort(["s1","q"])))
# 49 dropped: p2 in current scores
sc = pl.read_parquet(R + "work/frfix2/test_scores_v9b_fpveto.parquet", columns=["q", "s", "p2"]).join(
    gr.join(veto, on=["si","qi"], how="anti").select(s="si", q="qi"), on=["q","s"])
print("recall dropped pairs p2:", sc["p2"].describe().filter(pl.col("statistic").is_in(["count","min","mean","max"])).rows())
print("added by group:", dict(nc.join(add, on=["si","qi"]).group_by("grp").len().sort("grp").iter_rows()))
