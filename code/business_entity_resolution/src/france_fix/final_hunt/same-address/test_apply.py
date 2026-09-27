"""Apply the same-address generator to test: records unmatched in v10b, S1 of the same country. Evidence columns:
gathik agreement (pair in his v9 output, his p), our candidate-list membership and probability, veto-set membership."""
import os, sys, time
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salib import *
from common import WORK, id_to_int, log, read_tsv
OUT = rf"{SCRATCH}/final/same-address"
R = os.path.dirname(WORK)
t0 = time.time()
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
rec = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select("entity_id", "business_name", "business_address", "country") for k in (2, 3)])
         .select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address", country="country")
         .join(v.lazy().select("q").unique(), on="q", how="anti").collect())
log(f"v10b pairs {v.height}; unmatched test records {rec.height} {dict(rec.group_by('country').len().iter_rows())}  ({time.time()-t0:.0f}s)")
rec = add_key(rec, "qa")
s1k = add_key(s1, "sa")
log(f"keys built ({time.time()-t0:.0f}s)")
j = generate(rec.select("q", "qn", "qa", "country", "k"), s1k.select("s", "sn", "sa", "country", "k"))
log(f"candidates {j.height}; ok {j['ok'].sum()}; keep {j['keep'].sum()}  ({time.time()-t0:.0f}s)")
# evidence
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv")).with_columns(g_acc=pl.lit(True))
gp = i64(pl.read_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet"), columns=["q", "s", "p2"])).rename({"p2": "gp"})
lists = pl.concat([i64(pl.read_parquet(os.path.join(WORK, f), columns=["q", "s"])) for f in
                   ("test_scores_full_cons.parquet", os.path.join("ce_x", "test_rows.parquet"), "test_scores_full_cons_test_b2.parquet")]).unique().with_columns(in_list=pl.lit(True))
frp = i64(pl.read_parquet(os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet"), columns=["q", "s", "p2"])).rename({"p2": "fr_p"})
vetos = pl.concat([i64(pl.read_parquet(p).select("q", "s")) for p in (
    rf"{SCRATCH}/frfix/namechg/veto_set.parquet", os.path.join(WORK, "frfix2", "fp_veto_set.parquet"),
    os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet"))]).unique().with_columns(veto=pl.lit(True))
s1n = v.group_by("s").len("s1_nmatch")
j = (j.join(g, on=["q", "s"], how="left").join(gp, on=["q", "s"], how="left").join(lists, on=["q", "s"], how="left")
      .join(frp, on=["q", "s"], how="left").join(vetos, on=["q", "s"], how="left").join(s1n, on="s", how="left")
      .with_columns(pl.col("g_acc").fill_null(False), pl.col("in_list").fill_null(False), pl.col("veto").fill_null(False), pl.col("s1_nmatch").fill_null(0)))
# gathik matched this record to another S1?
gq = g.select("q", g_s="s").group_by("q").agg(pl.col("g_s").first())
j = j.join(gq, on="q", how="left").with_columns(g_other=pl.col("g_s").is_not_null() & (pl.col("g_s") != pl.col("s")))
j.write_parquet(os.path.join(OUT, "test_cands.parquet"))
k = j.filter(pl.col("keep"))
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
log(str(k.group_by("country").agg(n=pl.len(), veto=pl.col("veto").sum(), g_acc=pl.col("g_acc").mean(), g_other=pl.col("g_other").mean(),
                                  in_list=pl.col("in_list").mean(), s1_empty=(pl.col("s1_nmatch") == 0).mean(), low=pl.col("low").mean())))
k = k.with_columns(nm=pl.col("nops").str.extract_all(r"n_[a-z_]+").list.join("+"))
for c in ("France", "US", "India"):
    kc = k.filter(pl.col("country") == c)
    log(f"--- {c}\n" + str(kc.group_by("nm").agg(n=pl.len(), low=pl.col("low").mean(), g_acc=pl.col("g_acc").mean(), in_list=pl.col("in_list").mean(),
                                                 frp=pl.col("fr_p").mean()).sort("n", descending=True).head(30)))
log(f"done {time.time()-t0:.0f}s")
