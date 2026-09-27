import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT, WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import id_to_int
W = f"{WORK}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
fr = (pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(id_to_int("entity_id").alias("s")).collect())
if fr.height == 0:
    print(pl.scan_parquet(W+"test_s1.parquet").select(pl.col("country").value_counts()).collect())
print("France S1", fr.height)
sub = (pl.read_csv(f"{ROOT}/submissions/v10c/matching_results.tsv", separator="\t")
       .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids")!=""))
       .select(id_to_int("source1_entity_id").alias("s"), id_to_int("matched_entity_ids").alias("q")))
subfr = sub.join(fr, on="s", how="semi")
print("v10c France pairs", subfr.height, "rows w/ match", subfr["s"].n_unique())
subfr.write_parquet(OUT+"v10c_fr_pairs.parquet")
frs = fr["s"]
def load(f, cols, name):
    d = pl.scan_parquet(f).filter(pl.col("s").is_in(frs)).select(["q","s"]+cols).collect()
    print(name, "France rows", d.height, "records", d["q"].n_unique())
    return d
for f,n,c in [(W+"ce_b2/test_ce_bgef0.parquet","ce_b2 bgef0 raw",["p2","ce"]),(W+"ce_b2/test_scores_ce_bgefolds.parquet","ce_b2 bge scores",["p2"]),
            (W+"ce_b2/test_scores_ce_e5lfolds.parquet","ce_b2 e5l scores",["p2"]),(W+"ce_x/test_ce_bgef0.parquet","ce_x bgef0 raw",None),
            (W+"gathik/v9/ce_x_test_rows.parquet","gathik cc rows",["p2"]),(W+"gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet","gathik s3",["p2"]),
            (W+"test_scores_full_cons.parquet","ours gbdt old",["p2"]), (W+"test_scores_blend_v7p.parquet","v7p",["p2"])]:
    try:
        if c is None:
            print(f, pl.scan_parquet(f).collect_schema()); c=[x for x in pl.scan_parquet(f).collect_schema().names() if x not in("q","s")]
        d = load(f,c,n)
        j = subfr.join(d, on=["s","q"], how="left")
        print("   v10c France pairs covered:", j.filter(pl.col(c[0]).is_not_null()).height)
    except Exception as e: print(n, "ERR", e)
