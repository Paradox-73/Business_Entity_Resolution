"""Where is the remaining US/India held-out loss of the v10a-like blend (0.6 ours + 0.4 gathik where both, else either)?
Points = macro F0.5 gain on eval-half S1 if one mistake category were fixed. Writes nothing under work/."""
import os, sys, zlib
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, read_truth, id_to_int, macro_f05_df, log
from pipeline import decide_expf

G = os.path.join(WORK, "gathik", "v9")
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def held(oof_path, s3_path):
    oof = i64(pl.read_parquet(oof_path, columns=["q", "s", "p2"]))
    s3 = i64(pl.read_parquet(s3_path, columns=["q", "s", "p3"]))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", p=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0))
                    .otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))


s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country", "name_core", "addr_missing"])
s1 = s1.select(s=id_to_int("entity_id").cast(pl.Int64), country="country", s_core="name_core", s_noaddr="addr_missing",
               h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
s1 = s1.with_columns(chain=pl.len().over("country", "s_core"))
ev = s1.filter(pl.col("h") < 500)
truth_all = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))
truth = truth_all.join(ev.select("s"), on="s")
rec = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"), columns=["entity_id", "addr_missing", "name_core"]) for k in (2, 3)])
rec = rec.select(q=id_to_int("entity_id").cast(pl.Int64), q_noaddr="addr_missing", q_core="name_core")

sm = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_smallfolds.parquet"))
bg = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_bgefolds.parquet"))
ours = sm.rename({"p": "a"}).join(bg.rename({"p": "b"}), on=["q", "s"], how="full", coalesce=True)
ours = ours.select("q", "s", po=((0.3 * pl.col("a").fill_null(0) + 0.7 * pl.col("b").fill_null(0)) /
                   (pl.when(pl.col("a").is_null()).then(0.0).otherwise(0.3) + pl.when(pl.col("b").is_null()).then(0.0).otherwise(0.7))).cast(pl.Float32))
gat = held(os.path.join(G, "models_full_xgb_cons_oof.parquet"), os.path.join(G, "ce_x_oof_s3_bgef0bgef1bgef2.parquet")).rename({"p": "pg"})
d = ours.join(gat, on=["q", "s"], how="full", coalesce=True)
d = d.with_columns(p2=pl.when(pl.col("pg").is_null()).then(pl.col("po")).when(pl.col("po").is_null()).then(pl.col("pg"))
                   .otherwise(0.4 * pl.col("pg") + 0.6 * pl.col("po")))
del sm, bg, ours, gat
evs = ev.select("s")
pred = decide_expf(d.select("q", "s", "p2"), "p2", 0.5, 1.0).join(evs, on="s")
base = macro_f05_df(pred, truth, evs)
log(f"blend held-out {base:.5f} (expected 0.99205); eval S1 {evs.height}, truth pairs {truth.height}, pred pairs {pred.height}")

best = d.sort("p2", descending=True).unique("q", keep="first").select("q", bs="s", bp="p2")
cand = d.select("q", "s", "p2", "po", "pg")
tq = truth_all.group_by("q").agg(ts=pl.col("s").first())
fn = (truth.join(pred.with_columns(hit=pl.lit(True)), on=["s", "q"], how="left").filter(pl.col("hit").is_null()).drop("hit")
      .join(cand, on=["q", "s"], how="left").join(best, on="q", how="left"))
fn = fn.with_columns(cat=pl.when(pl.col("p2").is_null()).then(pl.lit("FN not in any list"))
                     .when(pl.col("bs") != pl.col("s")).then(pl.lit("FN record's best S1 is another"))
                     .otherwise(pl.lit("FN best S1 right, rule rejected")))
fp = (pred.join(truth.with_columns(y=pl.lit(True)), on=["s", "q"], how="left").filter(pl.col("y").is_null()).drop("y")
      .join(tq, on="q", how="left").join(cand, on=["q", "s"], how="left"))
fp = fp.with_columns(cat=pl.when(pl.col("ts").is_null()).then(pl.lit("FP record has no true S1 (decoy/unmatched)"))
                     .otherwise(pl.lit("FP record belongs to another S1")))

# per-S1 counts
cnt = (evs.join(pred.group_by("s").len("np"), on="s", how="left").join(truth.group_by("s").len("nt"), on="s", how="left")
          .join(pred.join(truth, on=["s", "q"]).group_by("s").len("tp"), on="s", how="left").fill_null(0))


def F(tp, np_, nt):
    p, r = pl.col(tp) / pl.col(np_), pl.col(tp) / pl.col(nt)
    return (pl.when(pl.col(nt) == 0).then((pl.col(np_) == 0).cast(pl.Float64)).when(pl.col(tp) == 0).then(0.0)
              .otherwise(1.25 * p * r / (0.25 * p + r)))


cnt = cnt.with_columns(f=F("tp", "np", "nt"))
N = evs.height


def gain(sub, kind):
    g = sub.group_by("s").len("k")
    c = cnt.join(g, on="s", how="left").with_columns(pl.col("k").fill_null(0))
    if kind == "fn":
        c = c.with_columns(tp2=pl.col("tp") + pl.col("k"), np2=pl.col("np") + pl.col("k"), nt2=pl.col("nt"))
    else:
        c = c.with_columns(tp2=pl.col("tp"), np2=pl.col("np") - pl.col("k"), nt2=pl.col("nt"))
    return c.select((F("tp2", "np2", "nt2") - pl.col("f")).sum()).item() / N


rows = []
fn = fn.join(ev.select("s", "country", "chain", "s_noaddr"), on="s").join(rec, on="q", how="left")
fp = fp.join(ev.select("s", "country", "chain", "s_noaddr"), on="s").join(rec, on="q", how="left")
log(f"total loss {1 - base:.5f}; S1 with F<1: {cnt.filter(pl.col('f') < 1).height} of {N}")
for kind, df in (("fn", fn), ("fp", fp)):
    for c in sorted(df["cat"].unique().to_list()):
        sub = df.filter(pl.col("cat") == c)
        line = f"{c:48s} n {sub.height:7d} pts {gain(sub, kind):.5f}"
        for ctry in ("US", "India"):
            line += f" | {ctry} {gain(sub.filter(pl.col('country') == ctry), kind):.5f}"
        line += f" | rec no-addr {gain(sub.filter(pl.col('q_noaddr')), kind):.5f} | chain>=2 {gain(sub.filter(pl.col('chain') >= 2), kind):.5f}"
        log(line)
        if c.startswith("FN best") or c.startswith("FP"):
            b = sub.with_columns(pb=(pl.col("p2") * 10).floor().clip(0, 9)).group_by("pb").agg(n=pl.len()).sort("pb")
            log("   p2 deciles: " + str(dict(b.iter_rows())))
# any-pipeline recall: pairs in ours-only / gathik-only lists
log(f"FN not in list but in ours only/gathik only: n/a (union used)")
# oracle: if each S1's predicted set were the best subset of the CANDIDATES (perfect ranking), upper bound
allfix = gain(pl.concat([fn.filter(pl.col("cat") != "FN not in any list").select("s", "q")]), "fn")
log(f"bound: all in-list FNs fixed {allfix:.5f}; all FPs removed {gain(fp.select('s', 'q'), 'fp'):.5f}")
fn.select("s", "q", "cat", "p2", "po", "pg", "bs", "bp", "country", "chain", "q_noaddr").write_parquet(r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/gap/fn.parquet")
fp.select("s", "q", "cat", "p2", "po", "pg", "ts", "country", "chain", "q_noaddr").write_parquet(r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/gap/fp.parquet")
log("done")
