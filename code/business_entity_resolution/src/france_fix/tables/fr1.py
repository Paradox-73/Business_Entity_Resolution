"""Best-candidate table of the France test records -> $BER_SCRATCH/france/fr_top.parquet.

Inputs: WORK/ce/test_scores_blend_ab_a2_frmin.parquet (France scores of README step 10), WORK/test_scores_full_cons.parquet
(GBDT p1, p2) and submissions/v7ens/matching_results.tsv. One row per record: its best and 2nd candidate by p2, their
texts, the change pattern `pat` (france_cal.py) and whether v7ens accepted the pair (`acc`).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
import france_cal as fc
OUT = f"{SCRATCH}/france"
t0 = time.time()
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(s=id_to_int("entity_id"), c="country")
frs = s1.filter(pl.col("c") == "France").select("s")
b = pl.read_parquet(os.path.join(WORK, "ce", "test_scores_blend_ab_a2_frmin.parquet"), columns=["q", "s", "p2"]).with_columns(
    pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(frs, on="s")
g = pl.read_parquet(os.path.join(WORK, "test_scores_full_cons.parquet"), columns=["q", "s", "p1", "p2"]).rename({"p2": "p2g"}).with_columns(
    pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(frs, on="s")
b = b.join(g, on=["q", "s"], how="left")
print("France scored pairs", b.height, "records", b["q"].n_unique(), time.time() - t0)
b = fc.attach_text(b, "test")
b = fc.veto_fr(b)
b = b.sort("p2", descending=True)
b = b.with_columns(rk=pl.int_range(1, pl.len() + 1).over("q"))
top = b.filter(pl.col("rk") == 1)
sec = b.filter(pl.col("rk") == 2).select("q", s_2="s", p2_2="p2", p1_2="p1", sn_2="sn", sa_2="sa", pat_2="pat")
top = top.join(sec, on="q", how="left")
sub = pl.read_csv(rf"{ROOT}/submissions/v7ens/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
pairs = (sub.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids")).join(frs, on="s"))
print("v7ens France pairs", pairs.height)
top = top.join(pairs.with_columns(acc=pl.lit(True)), on=["q", "s"], how="left").with_columns(pl.col("acc").fill_null(False))
top.drop("c", strict=False).write_parquet(f"{OUT}/fr_top.parquet")
print("wrote", top.height, time.time() - t0)
r = top.group_by("pat").agg(n=pl.len(), acc=pl.col("acc").mean(), p50=pl.col("p2").median(), p2g50=pl.col("p2g").median(),
                            sec_hi=(pl.col("p2_2").fill_null(0) >= 0.2).mean()).sort("n", descending=True)
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(r)
