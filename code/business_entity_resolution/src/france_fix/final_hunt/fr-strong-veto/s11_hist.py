import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import id_to_int
R = f"{ROOT}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
frs = pl.scan_parquet(R+"work/test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id")).collect()
def pairs(p):
    return (pl.read_csv(p, separator="\t").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids")!="").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids")).join(frs, on="s"))
ens = pairs(R+"submissions/v7ens/matching_results.tsv"); g = pairs(R+"work/out_v7g/matching_results.tsv"); m = pairs(R+"submissions/v7m/matching_results.tsv")
rem = ens.join(g, on=["s","q"], how="anti"); add = g.join(ens, on=["s","q"], how="anti")
mres = m.join(ens, on=["s","q"], how="anti")
print("v7ens FR", ens.height, "v7g FR", g.height, "v7g removed", rem.height, "added", add.height, "v7m restored", mres.height)
c = pl.read_parquet(OUT+"fr_cand_alt.parquet")
v = pl.read_parquet(OUT+"fr_v10c_scored.parquet")
for nm, x in [("v7g_removed", rem), ("v7m_restored", mres), ("v7ens", ens)]:
    k = c.join(x, on=["s","q"], how="semi")
    print(f"cands (both<0.5, n={c.height}) in {nm}: {k.height}; t<0.2 {k.filter((pl.col('ob')<0.2)&(pl.col('gb')<0.2)).height}")
print("cands in v7ens but not v7g-removed:", c.join(ens, on=["s","q"], how="semi").join(rem, on=["s","q"], how="anti").height)
print("cands not in v7ens:", c.join(ens, on=["s","q"], how="anti").height)
# where did v7g removed pairs go in v10c; their strong scores
rv = v.join(rem, on=["s","q"], how="semi")
print("v7g-removed pairs still in v10c:", rv.height, "with both cc", rv.filter(pl.col("occ")&pl.col("gcc")).height, "both<0.5", rv.filter(pl.col("occ")&pl.col("gcc")&(pl.col("ob")<0.5)&(pl.col("gb")<0.5)).height,
      "ob mean", rv["ob"].mean(), "gb mean", rv["gb"].mean())
mv = v.join(mres, on=["s","q"], how="semi")
print("v7m-restored pairs in v10c:", mv.height, "both cc", mv.filter(pl.col("occ")&pl.col("gcc")).height)
c.join(rem, on=["s","q"], how="semi").select("s","q").write_parquet(OUT+"cand_in_v7g_removed.parquet")
