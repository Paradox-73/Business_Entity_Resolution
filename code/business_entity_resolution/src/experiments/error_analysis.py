"""Error analysis on out-of-fold predictions saved by `pipeline.py train <tag>`.

Usage: python error_analysis.py <tag> [n_examples]

Mistake types (per S2/S3 row q whose true S1 is t, predicted top S1 p), eval S1 rows only:
  blocking_miss : true pair not in the shortlist
  under_thresh  : top candidate is correct but prob < threshold (missed match)
  wrong_s1      : top candidate is a different S1 (a miss, and a false merge if above threshold)
  decoy_merge   : q has no true S1 but got assigned (false merge)
Points lost by each type = score gain if only that type were fixed (oracle counterfactual).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import json
import os
import sys
import polars as pl
from common import WORK, log, macro_f05_df, int_to_id
from pipeline import train_meta, attach, chunk_files, decide, pairs_dir, SEG_COLS


def main(tag, n_ex=10):
    md = os.path.join(WORK, "models", tag)
    res = json.load(open(os.path.join(md, "result.json")))
    prob = "p2" if res["use_stage2"] else "p1"
    thr = res["thresholds"] if res["use_stage2"] else res["stage1_global"][0]
    B = pl.read_parquet(os.path.join(md, "oof.parquet"))
    s1, t, qmap = train_meta(tag)
    qinfo = pl.read_parquet(os.path.join(pairs_dir(tag), "q.parquet"))
    s1e = s1.filter("is_eval").select("s")
    te = t.join(s1e, on="s")
    found = pl.concat([attach(pl.read_parquet(f, columns=["q", "s"]), qmap).filter("label").select("q", "s") for f in chunk_files(tag)])
    pred = decide(B, prob, thr, qinfo.drop("country"))
    base = macro_f05_df(pred, te, s1e)
    log(f"{tag}: {prob}, macro F0.5 on eval S1 = {base:.4f}")

    top = (B.sort(prob, descending=True).unique("q", keep="first")
             .join(te.select("q", true_s="s"), on="q", how="left"))
    pq = pred.select("q", assigned=pl.lit(True))
    top = top.join(pq, on="q", how="left").with_columns(pl.col("assigned").fill_null(False))
    miss = te.join(found, on=["q", "s"], how="anti").select("q", true_s="s")
    under = top.filter(pl.col("label") & ~pl.col("assigned"))
    wrong = top.filter(pl.col("true_s").is_not_null() & ~pl.col("label"))
    decoy = top.filter(pl.col("true_s").is_null() & pl.col("assigned") & pl.col("s").is_in(s1e["s"]))

    def gain(add=None, remove=None):
        pm = pred
        if remove is not None:
            pm = pm.join(remove.select("s", "q"), on=["s", "q"], how="anti")
        if add is not None:
            pm = pl.concat([pm, add.select("s", "q")]).unique()
        return macro_f05_df(pm, te, s1e) - base

    g = {"blocking_miss": gain(add=miss.rename({"true_s": "s"})),
         "under_thresh": gain(add=under),
         "wrong_s1": gain(add=wrong.select(pl.col("true_s").alias("s"), "q"), remove=wrong.filter("assigned")),
         "decoy_merge": gain(remove=decoy)}
    counts = {"blocking_miss": miss.height, "under_thresh": under.height, "wrong_s1": wrong.height,
              "decoy_merge": decoy.height}
    log("POINTS LOST per mistake type (count):")
    for k, v in sorted(g.items(), key=lambda x: -x[1]):
        log(f"   {k:14s} +{v:.4f}  ({counts[k]})")
    json.dump({"score": base, "points": g, "counts": counts}, open(os.path.join(md, "errors.json"), "w"), indent=1)

    ref = te.join(qinfo, on="q")
    for name, d in (("blocking_miss", miss), ("under_thresh", under), ("wrong_s1", wrong), ("decoy_merge", decoy)):
        d = d.join(qinfo, on="q", how="left")
        log(f"=== {name}: {d.height}")
        for col in ["country"] + SEG_COLS:
            sh = dict(d[col].value_counts(normalize=True).iter_rows())
            rf = dict(ref[col].value_counts(normalize=True).iter_rows())
            log(f"   {col}: " + ", ".join(f"{a}={sh.get(a, 0):.1%} (all {rf[a]:.1%})" for a in sorted(rf, key=str)))
        if n_ex:
            show_examples(d, n_ex)


def show_examples(d, n):
    ex = d.sample(min(n, d.height), seed=0)
    raw = {}
    for k in (1, 2, 3):
        r = pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"), columns=["entity_id", "business_name", "business_address"])
        raw[k] = r
    allr = pl.concat(raw.values())
    ids = ex.with_columns(qid=int_to_id("q"), tid=int_to_id("true_s") if "true_s" in ex.columns else pl.lit(None),
                          pid=int_to_id("s") if "s" in ex.columns else pl.lit(None))
    look = dict((a, (b, c)) for a, b, c in allr.filter(pl.col("entity_id").is_in(
        pl.concat([ids["qid"], ids["tid"].drop_nulls(), ids["pid"].drop_nulls()]))).iter_rows())
    for r in ids.iter_rows(named=True):
        qn = look.get(r["qid"], ("?", "?"))
        print(f"   Q {str(qn[0])[:40]} | {str(qn[1])[:55]}")
        if r.get("tid"):
            tn = look.get(r["tid"], ("?", "?"))
            print(f"      TRUE {str(tn[0])[:40]} | {str(tn[1])[:55]}")
        if r.get("pid") and r.get("pid") != r.get("tid"):
            pn = look.get(r["pid"], ("?", "?"))
            pr = r.get("p2", r.get("p1"))
            print(f"      PRED {str(pn[0])[:40]} | {str(pn[1])[:55]}  p={pr:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
