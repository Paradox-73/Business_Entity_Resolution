"""US/India 'list-mover' correction: where the wide-candidate build and the old-candidate build disagree on a pair, the
decision goes to the transformer-only probability (pt), which does not depend on the candidate list.

  python movers/build_movers.py <wide_sub> <old_sub> <base_sub> <out_dir> [wide_scores old_scores]    (from src/)
  v9zm / v10a / v10d: python movers/build_movers.py v7p v7q v9y $BER_WORK/movers
       (submissions/<name>/matching_results.tsv; scores default WORK/test_scores_blend_<wide>.parquet / _<old>.parquet)

pt = mean over families (bge-reranker, e5-small) of P(true | mean 3-fold logit), calibrated per country on train
out-of-fold close calls (0.5-wide logit bins); logits from the wide (ce_b2) or old (ce_x) close-call scoring.
sc_w / sc_o: the pair was a transformer close call in the wide / old build.
  tier 1 remove : in base, wide-only pair (old build rejected it), S1 in the old list, wide build GBDT-only (no transformer),
                  old build scored it, pt < 0.5
  tier 1 restore: old-only pair (wide build rejected it), record unmatched in base, wide build GBDT-only or pair absent,
                  old build scored it, pt >= 0.8
  tier 2 remove : as tier 1 remove but both builds scored it, pt < 0.2
  tier 2 restore: as tier 1 restore but both builds scored it, pt >= 0.95
Writes <out_dir>/usi_<base>_tier1.parquet and usi_<base>_tier12.parquet (US/India pairs s, q) + the change lists."""
import os
import sys
import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/
from common import ROOT, WORK, id_to_int  # noqa: E402

wide, old, base, out = sys.argv[1:5]
ws = sys.argv[5] if len(sys.argv) > 5 else f"{WORK}/test_scores_blend_{wide}.parquet"
os_ = sys.argv[6] if len(sys.argv) > 6 else f"{WORK}/test_scores_blend_{old}.parquet"
os.makedirs(out, exist_ok=True)
s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"]).select(s=id_to_int("entity_id").cast(pl.Int64), country="country")


def pairs(name):
    """US/India (q, s, country) pairs of <root>/submissions/<name>/matching_results.tsv."""
    d = pl.read_csv(os.path.join(ROOT, "submissions", name, "matching_results.tsv"), separator="\t", quote_char=None, infer_schema_length=0) \
        .filter(pl.col("matched_entity_ids").fill_null("") != "")
    return d.select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=pl.col("matched_entity_ids").str.split(",")).explode("q") \
        .with_columns(q=id_to_int("q").cast(pl.Int64)).join(s1, on="s").filter(pl.col("country") != "France").select("q", "s", "country")


Wd, O, B = pairs(wide), pairs(old), pairs(base)
qs = pl.concat([Wd.select("q"), O.select("q")]).unique()
sw = pl.read_parquet(ws, columns=["q", "s", "p2"]).join(qs, on="q").rename({"p2": "pw"})
so = pl.read_parquet(os_, columns=["q", "s", "p2"]).join(qs, on="q").rename({"p2": "po"})
best_o = so.sort("po", descending=True).unique("q", keep="first").select("q", best_o="s")
cw = pl.read_parquet(f"{WORK}/ce_b2/test_rows.parquet", columns=["q", "s"]).with_columns(sc_w=pl.lit(True))
co = pl.read_parquet(f"{WORK}/ce_x/test_rows.parquet", columns=["q", "s"]).with_columns(sc_o=pl.lit(True))
cuts = list(np.arange(-8, 8.01, 0.5))


def fam_pt(fam):
    """For one family: the mean 3-fold logit of every test close-call pair (wide build ce_b2 first, then the
    old build ce_x), and P(true) per country and 0.5-wide logit bin from the train out-of-fold scores."""
    tr = pl.concat([pl.read_parquet(f"{WORK}/ce_b2/train_ce_{fam}f{k}.parquet", columns=["label", "country", "ce"]) for k in range(3)]) \
        .filter(pl.col("country") != "France")
    cal = tr.with_columns(b=pl.col("ce").cut(cuts)).group_by("country", "b").agg(pt=pl.col("label").mean())
    L = []
    for d in ("ce_b2", "ce_x"):
        te = None
        for k in range(3):
            x = pl.read_parquet(f"{WORK}/{d}/test_ce_{fam}f{k}.parquet", columns=["q", "s", "ce"]).rename({"ce": f"c{k}"})
            te = x if te is None else te.join(x, on=["q", "s"])
        L.append(te.with_columns(ce=(pl.col("c0") + pl.col("c1") + pl.col("c2")) / 3).select("q", "s", "ce"))
    return pl.concat(L).unique(["q", "s"], keep="first"), cal


def attach(d):
    """Add to pairs d the calibrated probabilities pt_bge, pt_small, their mean pt, and sc_w / sc_o (the pair
    was a close call of the wide / old build)."""
    for f in ("bge", "small"):
        L, cal = fam_pt(f)
        d = d.join(L.rename({"ce": f"ce_{f}"}), on=["q", "s"], how="left").with_columns(b=pl.col(f"ce_{f}").cut(cuts)) \
            .join(cal.rename({"pt": f"pt_{f}"}), on=["country", "b"], how="left").drop("b")
    return d.with_columns(pt=pl.mean_horizontal("pt_bge", "pt_small")) \
        .join(cw, on=["q", "s"], how="left").join(co, on=["q", "s"], how="left") \
        .with_columns(pl.col("sc_w").fill_null(False), pl.col("sc_o").fill_null(False))


adds = attach(Wd.join(O, on=["q", "s"], how="anti").join(sw, on=["q", "s"], how="left").join(so, on=["q", "s"], how="left")
              .filter(pl.col("po").is_not_null()))                         # S1 was in the old list too
drops = attach(O.join(Wd, on=["q", "s"], how="anti").join(sw, on=["q", "s"], how="left").join(so, on=["q", "s"], how="left")
               .join(best_o, on="q", how="left").filter(pl.col("best_o") == pl.col("s"))
               .join(B.select("q"), on="q", how="anti"))                      # record unmatched in the base file
adds = adds.join(B.select("q", "s"), on=["q", "s"], how="semi")              # only pairs the base file still has
rm1 = adds.filter(~pl.col("sc_w") & pl.col("sc_o") & (pl.col("pt") < 0.5))
rs1 = drops.filter(~pl.col("sc_w") & pl.col("sc_o") & (pl.col("pt") >= 0.8))
rm2 = adds.filter(pl.col("sc_w") & pl.col("sc_o") & (pl.col("pt") < 0.2))
rs2 = drops.filter(pl.col("sc_w") & pl.col("sc_o") & (pl.col("pt") >= 0.95))
print(f"wide-only pairs with S1 in old list {adds.height}, old-only same-best pairs on unmatched records {drops.height}")
print(f"tier 1: remove {rm1.height}, restore {rs1.height}; tier 2: remove {rm2.height}, restore {rs2.height}")
cols = ["q", "s", "country", "pw", "po", "pt", "pt_bge", "pt_small", "sc_w", "sc_o"]
for nm, rm, rs in (("tier1", rm1, rs1), ("tier12", pl.concat([rm1, rm2]), pl.concat([rs1, rs2]))):
    res = pl.concat([B.join(rm.select("q", "s"), on=["q", "s"], how="anti").select("s", "q"), rs.select("s", "q")])
    assert res["q"].n_unique() == res.height, "a record matched twice"
    res.write_parquet(os.path.join(out, f"usi_{base}_{nm}.parquet"))
    rm.select(cols).write_parquet(os.path.join(out, f"{nm}_removed.parquet"))
    rs.select(cols).write_parquet(os.path.join(out, f"{nm}_restored.parquet"))
    print(f"{nm}: base {B.height} -> {res.height} US/India pairs (-{rm.height} +{rs.height}); "
          f"by country removed {dict(rm.group_by('country').len().rows())} restored {dict(rs.group_by('country').len().rows())}")
