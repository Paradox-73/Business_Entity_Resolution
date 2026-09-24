"""Error analysis on out-of-fold predictions of the dev slice.

Usage: python error_analysis.py [path_to_dev_pairs_with_prob]
If the pairs file has no `prob` column, OOF predictions are recomputed with pipeline.PARAMS.

Mistake types (per S2/S3 row q whose true S1 is t, predicted S1 p):
  blocking_miss : true pair not in the shortlist
  under_thresh  : top candidate is correct but prob < threshold (missed match)
  wrong_s1      : top candidate is a different S1 (costs a miss AND, if above threshold, a false merge)
  decoy_merge   : q has no true S1 but got assigned (false merge)
Points lost by each type = score gain if only that type were fixed (oracle counterfactual).
"""
import os
import sys
import zlib
import numpy as np
import polars as pl
import lightgbm as lgb
from common import WORK, log, read_truth, macro_f05
from candidates import FEATURES
from pipeline import PARAMS, ROUNDS, TRAIN_ROWS, N_FOLDS, Xmat


def oof_predict(P):
    oof = np.zeros(P.height, np.float32)
    for f in range(N_FOLDS):
        trn = P.filter(pl.col("fold") != f)
        if trn.height > TRAIN_ROWS:
            trn = trn.sample(TRAIN_ROWS, seed=f)
        m = lgb.train(PARAMS, lgb.Dataset(Xmat(trn), trn["label"].to_numpy()), ROUNDS)
        mask = (P["fold"] == f).to_numpy()
        oof[mask] = m.predict(Xmat(P.filter(pl.col("fold") == f)))
        log(f"fold {f}")
    return P.with_columns(prob=pl.Series(oof))


def to_map(df, s="s_id", q="q_id"):
    out = {}
    for a, b in df.select(s, q).iter_rows():
        out.setdefault(a, set()).add(b)
    return out


def main(path=None, t=None):
    path = path or os.path.join(WORK, "dev_pairs.parquet")
    P = pl.read_parquet(path)
    if "prob" not in P.columns:
        P = oof_predict(P)
        P.write_parquet(path)
    frac = 0.1
    truth = read_truth()
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    s1 = s1.filter(pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 < frac * 1000,
                                                   return_dtype=pl.Boolean))
    s1_ids = s1["entity_id"].to_list()
    tr = truth.filter(pl.col("s1_id").is_in(s1_ids))
    true_map = to_map(tr, "s1_id", "q_id")
    tq = dict(zip(tr["q_id"], tr["s1_id"]))

    if t is None:
        grid = np.arange(0.5, 0.95, 0.025)
        t = max(grid, key=lambda x: macro_f05(to_map(P.sort("prob", descending=True).unique("q_id", keep="first")
                                                      .filter(pl.col("prob") >= x)), true_map, s1_ids))
    top = P.sort("prob", descending=True).unique("q_id", keep="first")
    pred = top.filter(pl.col("prob") >= t)
    base = macro_f05(to_map(pred), true_map, s1_ids)
    log(f"threshold {t:.3f}  macro F0.5 {base:.4f}")

    # classify every q with a true S1 or a prediction
    top = top.with_columns(true_s=pl.col("q_id").replace_strict(tq, default=None))
    in_short = set(P.filter("label")["q_id"])
    trq = pl.DataFrame({"q_id": list(tq.keys()), "true_s": list(tq.values())})
    miss = trq.filter(~pl.col("q_id").is_in(list(in_short)))
    under = top.filter(pl.col("label") & (pl.col("prob") < t))
    wrong = top.filter(pl.col("true_s").is_not_null() & ~pl.col("label"))
    decoy = top.filter(pl.col("true_s").is_null() & (pl.col("prob") >= t))
    log(f"counts: blocking_miss {miss.height}, under_thresh {under.height}, wrong_s1 {wrong.height} "
        f"(above t: {wrong.filter(pl.col('prob') >= t).height}), decoy_merge {decoy.height}; true pairs {len(tq)}")

    def score_with(add=(), remove=()):
        pm = {k: set(v) for k, v in to_map(pred).items()}
        for s, q in remove:
            pm.get(s, set()).discard(q)
        for s, q in add:
            pm.setdefault(s, set()).add(q)
        return macro_f05(pm, true_map, s1_ids)

    gains = {
        "blocking_miss": score_with(add=[(s, q) for q, s in miss.select("q_id", "true_s").iter_rows()]),
        "under_thresh": score_with(add=[(s, q) for s, q in under.select("s_id", "q_id").iter_rows()]),
        "wrong_s1": score_with(add=[(s, q) for s, q in wrong.select("true_s", "q_id").iter_rows()],
                               remove=[(s, q) for s, q in wrong.filter(pl.col("prob") >= t).select("s_id", "q_id").iter_rows()]),
        "decoy_merge": score_with(remove=[(s, q) for s, q in decoy.select("s_id", "q_id").iter_rows()]),
    }
    log("POINTS LOST per mistake type (score if fixed - current):")
    for k, v in sorted(gains.items(), key=lambda x: -x[1]):
        log(f"   {k:14s} +{v - base:.4f}")

    # breakdown by record type
    q = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet")) for k in (2, 3)])
    q = q.filter(pl.col("entity_id").is_in(list(set(P["q_id"]) | set(miss["q_id"]))))
    qi = q.select(pl.col("entity_id").alias("q_id"), "country", "src", "name_nonlatin", "name_is_domain",
                  "name_has_alias", "addr_missing", "addr_nonlatin", "business_name", "business_address")
    s1i = s1.select(pl.col("entity_id").alias("s_id"), pl.col("business_name").alias("s_name"),
                    pl.col("business_address").alias("s_addr"))
    tq_df = trq.join(qi, on="q_id")
    for name, d in (("blocking_miss", miss), ("under_thresh", under), ("wrong_s1", wrong), ("decoy_merge", decoy)):
        d = d.join(qi, on="q_id", how="left")
        log(f"\n=== {name}: {d.height} rows")
        for col in ("country", "src", "name_nonlatin", "name_is_domain", "addr_missing", "addr_nonlatin"):
            share = dict(d[col].value_counts(normalize=True).iter_rows())
            ref = dict((tq_df if name != "decoy_merge" else qi)[col].value_counts(normalize=True).iter_rows())
            log(f"   {col}: " + ", ".join(f"{a}={share.get(a, 0):.1%} (all {ref[a]:.1%})" for a in sorted(ref, key=str)))
        ex = d.sample(min(12, d.height), seed=0)
        if "s_id" in ex.columns:
            ex = ex.join(s1i, on="s_id", how="left")
        if "true_s" in ex.columns:
            ex = ex.join(s1i.rename({"s_id": "true_s", "s_name": "t_name", "s_addr": "t_addr"}), on="true_s", how="left")
        for r in ex.iter_rows(named=True):
            line = f"   Q[{r['country']}] {str(r['business_name'])[:40]} | {str(r['business_address'])[:50]}"
            if r.get("t_name"):
                line += f"\n      TRUE  {str(r['t_name'])[:40]} | {str(r['t_addr'])[:50]}"
            if r.get("s_name"):
                line += f"\n      PRED  {str(r['s_name'])[:40]} | {str(r['s_addr'])[:50]}  p={r.get('prob', 0):.2f}"
            print(line)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
