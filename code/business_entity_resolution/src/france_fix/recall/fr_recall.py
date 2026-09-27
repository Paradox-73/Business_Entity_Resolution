"""France pairs Gathik v9 matched that our France candidate lists never had, for records v10a leaves unmatched.
Profile them (edit ops, house-number direction, lowercase artifact) next to the US/India equivalents already in v9y."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/census"))
import polars as pl
from common import WORK, id_to_int, log, read_tsv
from ops import ops

R = os.path.dirname(WORK)
T = rf"{SCRATCH}/gap"


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
rec = pl.concat([pl.read_parquet(os.path.join(WORK, f"test_s{k}.parquet"), columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).select(
    q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address")
lists = pl.concat([pl.read_parquet(os.path.join(WORK, "test_scores_full_cons.parquet"), columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)),
                   pl.read_parquet(os.path.join(WORK, "ce_x", "test_rows.parquet"), columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)),
                   pl.read_parquet(os.path.join(WORK, "test_scores_full_cons_test_b2.parquet"), columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))]).unique()
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv")).join(s1.select("s", "country"), on="s")
v = pairs(os.path.join(R, "submissions", "v10a", "matching_results.tsv")).join(s1.select("s", "country"), on="s")
log(f"gathik pairs {dict(g.group_by('country').len().iter_rows())}; v10a pairs {dict(v.group_by('country').len().iter_rows())}")
gp = pl.read_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet"), columns=["q", "s", "p2"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
out = {}
for ctry in ("France", "US", "India"):
    gc = g.filter(pl.col("country") == ctry)
    notlist = gc.join(lists, on=["q", "s"], how="anti")
    free = notlist.join(v.select("q"), on="q", how="anti")
    s1free = free.join(v.select("s").unique(), on="s", how="anti")
    log(f"{ctry}: gathik pairs {gc.height}; not in our lists {notlist.height}; of those record unmatched in v10a {free.height} "
        f"(S1 row empty in v10a: {s1free.height})")
    out[ctry] = free
fr = out["France"].join(s1, on="s").join(rec, on="q").join(gp, on=["q", "s"], how="left")
us = pl.concat([out["US"], out["India"]]).join(s1, on="s").join(rec, on="q").join(gp, on=["q", "s"], how="left")


def prof(d, name):
    rows = []
    for r in d.iter_rows(named=True):
        o = ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or "")
        rows.append(";".join(sorted(o)))
    d = d.with_columns(ops=pl.Series(rows), low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
    top = (d.with_columns(nm=pl.col("ops").str.extract_all(r"n_[a-z_]+(?::[a-z]+)?").list.join("+"),
                          num=pl.col("ops").str.extract(r"(a_num[a-z_]*)"))
             .group_by("nm", "num").agg(n=pl.len(), low=pl.col("low").mean(), p=pl.col("p2").mean()).sort("n", descending=True))
    log(f"{name}: {d.height} pairs; lowercase share {d['low'].mean():.4f}; gathik p mean {d['p2'].mean():.3f}\n" + str(top.head(25)))
    return d


pd_ = prof(fr, "FRANCE gathik-only pairs")
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(60)
log("examples\n" + str(pd_.select("qn", "qa", "sn", "sa", "p2").head(25)))
pd_.write_parquet(os.path.join(T, "fr_gathik_free.parquet"))
pu = prof(us.sample(min(us.height, 20000), seed=1), "US/India gathik-only pairs (these were added in v9y)")
