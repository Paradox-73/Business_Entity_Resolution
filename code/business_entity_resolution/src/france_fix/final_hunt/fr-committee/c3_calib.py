"""Calibrate the lowercase test on France: v10b France matched pairs (mostly true) and veto-set pairs (mostly decoys),
split word-change / no-word-change; compare with committee add candidates."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from common import WORK, id_to_int, log, read_tsv
from ops import ops
R = os.path.dirname(WORK)
T = rf"{SCRATCH}/final/fr-committee"
TMP = rf"{SCRATCH}"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address").filter(pl.col("country") == "France").drop("country")
v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv")).join(s1.select("s"), on="s").sample(60000, seed=3).with_columns(src=pl.lit("v10b_fr"))
vt = pl.concat([
    pl.read_parquet(os.path.join(TMP, "frfix", "namechg", "veto_set.parquet"), columns=["q", "s"]).with_columns(src=pl.lit("veto_desc")),
    pl.read_parquet(os.path.join(WORK, "frfix2", "fp_veto_set.parquet"), columns=["q", "s"]).with_columns(src=pl.lit("veto_fp")),
]).pipe(i64).join(s1.select("s"), on="s")
d = pl.concat([v.select("s", "q", "src"), vt.select("s", "q", "src")])
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select("entity_id", "business_name", "business_address") for k in (2, 3)]
                ).select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address").join(d.select("q").unique().lazy(), on="q").collect()
d = d.join(s1, on="s").join(rec, on="q")
d = d.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or ""))) for r in d.iter_rows(named=True)], dtype=pl.Utf8),
                   low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
c = pl.read_parquet(os.path.join(T, "cand.parquet")).filter(pl.col("kind") == "add").select("s", "q", "ops", "low", "qn", "sn", "qa", "sa").with_columns(src=pl.lit("committee_add"))
d = pl.concat([d.select(c.columns), c])
d = d.with_columns(lowok=~pl.col("ops").str.contains("n_domain|n_squash"),
                   cls=pl.when(pl.col("ops").str.contains("n_swap:desc|n_add:desc")).then(pl.lit("desc"))
                   .when(pl.col("ops").str.contains("n_swap:|n_add:|n_drop:")).then(pl.lit("word_other"))
                   .when(pl.col("ops").str.contains("n_legal_change|n_legal_add")).then(pl.lit("legal"))
                   .otherwise(pl.lit("noword")),
                   up=pl.col("ops").str.contains("a_num_up"), down=pl.col("ops").str.contains("a_num_down"))
d.write_parquet(os.path.join(T, "calib.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
print(d.group_by("src", "cls").agg(n=pl.len(), low=pl.col("low").filter(pl.col("lowok")).mean(), up=pl.col("up").mean(), down=pl.col("down").mean()).sort("cls", "src"))
