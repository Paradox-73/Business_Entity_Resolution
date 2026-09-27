"""Where the held-out points go, for a trained model directory (expected-F0.5 decision as used for test).

  python error2.py <model_dir> <data_tag> [n_examples]      e.g. error2.py full_cons full 12

Per eval S1 row the macro F0.5 is computed; the loss (1 - F) is split over the S1 row's mistakes:
  miss_notop2   : true record never reached stage 2 (not shortlisted, or not in its top-2 by stage 1)
  miss_wrong_s1 : true record's best stage-2 candidate is another S1
  miss_under    : best candidate is the true S1 but the decision left it out
  fp_decoy      : record with no true S1 was assigned here
  fp_other      : record belonging to another S1 was assigned here
Each S1 row's loss is shared equally among its mistakes; totals are then summed by type and by segment
(no address, non-Latin name, country, house-number relation). Uses only the saved OOF table (light).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import json
import os
import sys
import polars as pl
from common import WORK, log, id_to_int
from pipeline import train_meta, pairs_dir, apply_decision

md, tag = os.path.join(WORK, "models", sys.argv[1]), sys.argv[2]
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 10
res = json.load(open(os.path.join(md, "result.json")))
B = pl.read_parquet(os.path.join(md, "oof.parquet"))
s1, t, _ = train_meta(tag)
qinfo = pl.read_parquet(os.path.join(pairs_dir(tag), "q.parquet"))
s1e = s1.filter("is_eval").select("s", "country")
te = t.join(s1e.select("s"), on="s")
M = apply_decision(B, "p2", res["decision"], qinfo.drop("country")).select("s", "q", pred=pl.lit(True))

# per-S1 F0.5
tp = M.join(te, on=["s", "q"]).group_by("s").len("tp")
d = (s1e.join(M.group_by("s").len("np"), on="s", how="left").join(te.group_by("s").len("nt"), on="s", how="left")
        .join(tp, on="s", how="left").fill_null(0))
p_, r_ = pl.col("tp") / pl.col("np"), pl.col("tp") / pl.col("nt")
f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
       .when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * p_ * r_ / (0.25 * p_ + r_)))
d = d.with_columns(F=f, loss=1 - f)
log(f"macro F0.5 on {d.height} eval S1: {d['F'].mean():.5f}  (total loss {d['loss'].mean():.5f})")

# mistakes
best = B.sort("p2", descending=True).unique("q", keep="first").select("q", best_s="s", best_p2="p2")
inB = B.select("q", "s", "p1", "p2")
fn = te.join(M, on=["s", "q"], how="anti").join(best, on="q", how="left").join(inB, on=["q", "s"], how="left")
fn = fn.with_columns(kind=pl.when(pl.col("p2").is_null()).then(pl.lit("miss_notop2"))
                         .when(pl.col("best_s") != pl.col("s")).then(pl.lit("miss_wrong_s1"))
                         .otherwise(pl.lit("miss_under")))
truth_q = t.select("q", true_s="s")
fp = (M.join(te, on=["s", "q"], how="anti").join(s1e.select("s"), on="s")
        .join(truth_q, on="q", how="left")
        .with_columns(kind=pl.when(pl.col("true_s").is_null()).then(pl.lit("fp_decoy")).otherwise(pl.lit("fp_other"))))
mist = pl.concat([fn.select("s", "q", "kind"), fp.select("s", "q", "kind")])
mist = mist.join(d.select("s", "loss"), on="s").with_columns(share=pl.col("loss") / pl.len().over("s"))
N = d.height
by_kind = mist.group_by("kind").agg(n=pl.len(), points=(pl.col("share").sum() / N)).sort("points", descending=True)
log("loss by mistake type (points of macro F0.5):")
for k, n, pts in by_kind.iter_rows():
    log(f"   {k:14s} {n:8d} mistakes  {pts:.5f}")

# segments
need_q, need_s = mist.select("q").unique().lazy(), mist.select("s").unique()
q2 = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet")).select(
        "entity_id", "addr_missing", "name_nonlatin", "addr_nums", "business_name", "business_address") for k in (2, 3)])
        .select(q=id_to_int("entity_id"), no_addr="addr_missing", nonlatin="name_nonlatin",
                qn1=pl.col("addr_nums").list.first(), qname="business_name", qaddr="business_address")
        .join(need_q, on="q").collect())
S1all = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "name_core"]).select(
    s=id_to_int("entity_id"), score_name="name_core")
S1t = (pl.scan_parquet(os.path.join(WORK, "train_s1.parquet")).select("entity_id", "addr_nums", "business_name", "business_address")
         .select(s=id_to_int("entity_id"), sn1=pl.col("addr_nums").list.first(), sname="business_name", saddr="business_address")
         .join(pl.concat([need_s, best.select(s="best_s").unique()]).unique().lazy(), on="s").collect()
         .join(S1all, on="s"))
chain = S1all.group_by("score_name").len("chain")
S1t = S1t.join(chain, on="score_name")
mist = mist.join(q2, on="q", how="left").join(S1t, on="s", how="left").join(s1e, on="s", how="left")
mist = mist.with_columns(numrel=pl.when(pl.col("qn1").is_null() | pl.col("sn1").is_null()).then(pl.lit("num_missing"))
                         .when(pl.col("qn1") == pl.col("sn1")).then(pl.lit("num_same"))
                         .when(pl.col("qn1").str.starts_with(pl.col("sn1")) | pl.col("sn1").str.starts_with(pl.col("qn1"))).then(pl.lit("num_prefix"))
                         .otherwise(pl.lit("num_diff")),
                         chainb=pl.when(pl.col("chain") <= 1).then(pl.lit("chain1")).when(pl.col("chain") <= 5).then(pl.lit("chain2-5")).otherwise(pl.lit("chain6+")))
for seg in ("no_addr", "nonlatin", "country", "numrel", "chainb"):
    g = mist.group_by("kind", seg).agg(points=(pl.col("share").sum() / N), n=pl.len()).sort("points", descending=True).head(10)
    log(f"by {seg}: " + "; ".join(f"{a}/{b}: {p:.5f} ({n})" for a, b, p, n in g.iter_rows()))

pl.Config.set_fmt_str_lengths(80)
for k in by_kind["kind"].to_list():
    ex = mist.filter(pl.col("kind") == k).sample(min(n_ex, mist.filter(pl.col("kind") == k).height), seed=1)
    log(f"--- examples {k}")
    for r in ex.iter_rows(named=True):
        wrong = ""
        if k in ("miss_wrong_s1",):
            bs = best.filter(pl.col("q") == r["q"]).row(0, named=True)
            w = S1t.filter(pl.col("s") == bs["best_s"])
            if w.height:
                w = w.row(0, named=True)
                wrong = f"\n        chosen: {w['sname']} | {w['saddr']} (p2 {bs['best_p2']:.2f})"
        log(f"   S1: {r['sname']} | {r['saddr']}  [chain {r['chain']}]\n        rec: {r['qname']} | {r['qaddr']}{wrong}")
