"""Accepted France pairs in v9y, joined with best-candidate names, ops and word diffs."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int, read_tsv
FR = f"{SCRATCH}/france"
NC = f"{SCRATCH}/frfix/namechg"
OUT = f"{SCRATCH}/frfix3/moreveto"
frs = pl.scan_parquet(f"{WORK}/test_s1.parquet").filter(pl.col("country") == "France").select(s=id_to_int("entity_id")).collect()
sub = read_tsv(f"{ROOT}/submissions/v9y/matching_results.tsv").with_columns(pl.col("matched_entity_ids").fill_null(""))
v = (sub.with_columns(s=id_to_int("source1_entity_id")).join(frs, on="s").with_columns(pl.col("matched_entity_ids").str.split(","))
     .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select("s", q=id_to_int("matched_entity_ids")))
print("v9y France pairs", v.height, "S1 rows", frs.height)
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "p2", "p2g", "qn", "qa", "sn", "sa", "pat", "acc", "s_2", "p2_2"])
v = v.join(t, on=["q", "s"], how="left")
print("not a fr_top best candidate:", v["qn"].null_count())
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q", "s", "ops", "nc", "num"])
v = v.join(o, on=["q", "s"], how="left")
c = pl.read_parquet(f"{NC}/fr_chg.parquet", columns=["q", "s", "p3", "added", "dropped", "samestreet", "twin_samenum"])
v = v.join(c, on=["q", "s"], how="left")
vs = pl.read_parquet(f"{NC}/veto_set.parquet", columns=["q", "s"]).with_columns(inveto=pl.lit(True))
v = v.join(vs, on=["q", "s"], how="left").with_columns(pl.col("inveto").fill_null(False))
print("in descriptor veto set but accepted in v9y:", v["inveto"].sum())
v.write_parquet(f"{OUT}/acc_v9y.parquet")
print(v.group_by(pl.col("pat").str.split("|").list.first().alias("kind"), pl.col("pat").str.split("|").list.last().alias("num")).len().sort("len", descending=True).head(30))
print(v.group_by("num").len().sort("len", descending=True))
