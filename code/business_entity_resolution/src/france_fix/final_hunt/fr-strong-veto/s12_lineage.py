import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import id_to_int
R = f"{ROOT}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
i64 = lambda d: d.select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64))
frs = pl.scan_parquet(R+"work/test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id")).collect()
def pairs(p):
    return (pl.read_csv(p, separator="\t").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids")!="").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids")).join(frs, on="s"))
c = pl.read_parquet(OUT+"fr_cand_alt.parquet")
V = {k: pairs(R+f"submissions/{k}/matching_results.tsv") for k in ["v7ens","v7m","v9y","v9zm","v10a","v10b"]}
V["v7g"] = pairs(R+"work/out_v7g/matching_results.tsv")
sets = {"recall_add_set(v9z?)": i64(pl.read_parquet(R+"work/frfix3/recall_add_set.parquet")), "v10b_added": i64(pl.read_parquet(R+"work/out_v10b_fr/france_recall_added.parquet")),
        "fr_amp": i64(pl.read_parquet(R+"work/final_sets/fr_amp_safe.parquet")), "fr_sameaddr": i64(pl.read_parquet(R+"work/final_sets/fr_same_address_safe.parquet")),
        "fr_typo": i64(pl.read_parquet(R+"work/final_sets/fr_typo_safe.parquet"))}
for k, x in list(V.items()) + list(sets.items()):
    c = c.with_columns(pl.struct("s","q").is_in(x.select(pl.struct("s","q")).to_series()).alias("in_"+k))
print("set sizes:", {k: x.height for k, x in sets.items()})
cols = [x for x in c.columns if x.startswith("in_")]
print(c.select([pl.col(x).sum() for x in cols]))
print(c.group_by(["in_v7ens","in_v7m","in_v9y","in_v9zm"]).agg(n=pl.len(), t02=((pl.col("ob")<0.2)&(pl.col("gb")<0.2)).sum()).sort("n", descending=True))
c.write_parquet(OUT+"fr_cand_lineage.parquet")
