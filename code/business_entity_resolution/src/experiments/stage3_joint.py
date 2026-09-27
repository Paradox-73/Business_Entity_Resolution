"""Stage 3 with several transformer families as features at once (instead of one stage 3 per family + blend.py).

Each family contributes its logit ce_<fam> plus its margin to the best other candidate and its rank within the
record; the GBDT features (p1, p2, p2_margin, n_cc, p1_rank) are shared. Same 3 folds, rules and "halves" protocol as
rerank.stage3, so the held-out number compares with blend.py's (27 Sep: small+bge blend 0.98878 halves / 0.98950 all).

  BER_CE_DIR=<ce dir> BER_S3_EXTRA=1 python stage3_joint.py <tag> small,bge [test_scores for BER_S3_TEST]
Writes <ce dir>/oof_s3_<tag>.parquet, test_scores_ce_<tag>.parquet, rule_<tag>.json (blend.py / finalize.py formats).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import json
import os
import sys
import numpy as np
import polars as pl
import rerank as R
from common import WORK, log, read_truth, id_to_int


def feats_for(r, fams):
    """rerank._s3_features per family on column ce_<fam> (EXTRA-style margins: best other candidate)."""
    r = r.with_columns(n_cc=pl.len().over("q").cast(pl.Float32),
                       p1_rank=pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Float32))
    for c in ["p2"] + [f"ce_{f}" for f in fams]:
        mx = pl.col(c).max().over("q")
        second = pl.col(c).drop_nulls().top_k(2).min().over("q")
        other = pl.when(pl.col(c).count().over("q") > 1).then(pl.when(pl.col(c) == mx).then(second).otherwise(mx)).otherwise(None)
        r = r.with_columns((pl.col(c) - other).alias(f"{c}_margin"))
        if c != "p2":
            r = r.with_columns(pl.col(c).rank("ordinal", descending=True).over("q").cast(pl.Float32).alias(f"{c}_rank"))
    r = r.with_columns(ce_mean=pl.mean_horizontal([f"ce_{f}" for f in fams]))
    names = ["p1", "p2", "p2_margin", "n_cc", "p1_rank", "ce_mean"] + [x for f in fams for x in (f"ce_{f}", f"ce_{f}_margin", f"ce_{f}_rank")]
    return r, names


def load(split, fams, fold=0):
    """Rows of `split` with one ce column per family."""
    out = None
    for f in fams:
        parts = [pl.read_parquet(os.path.join(R.OUT, f"{split}_ce_{f}f{k}.parquet")).with_columns(
            pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)) for k in range(3)]
        # train: each family's out-of-fold score; test: fold `fold`'s score (stage 3 is applied per fold, then averaged)
        d = pl.concat(parts) if split == "train" else parts[fold]
        d = d.rename({"ce": f"ce_{f}"})
        out = d if out is None else out.join(d.select("q", "s", f"ce_{f}"), on=["q", "s"])
    return out


def main(tagx, fams, test_scores=None):
    import xgboost as xgb
    params = dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.05, max_depth=6,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.9, seed=7)
    md = json.load(open(os.path.join(R.OUT, "select.json")))["model_dir"]
    tag = json.load(open(os.path.join(WORK, "models", md, "result.json")))["tag"]
    r, feats = feats_for(load("train", fams), fams)
    log(f"joint stage 3 on {fams}: {r.height} train rows, features {feats}")
    p3 = np.zeros(r.height, np.float32)
    for f in range(3):
        tr = r.filter(pl.col("fold") != f)
        m = xgb.train(params, xgb.DMatrix(tr.select(feats).to_numpy(), tr["label"].to_numpy()), 300)
        msk = (r["fold"] == f).to_numpy()
        p3[msk] = m.predict(xgb.DMatrix(r.filter(pl.Series(msk)).select(feats).to_numpy()))
    r = r.with_columns(p3=pl.Series(p3))
    r.select("q", "s", "p3", "label", "fold").write_parquet(os.path.join(R.OUT, f"oof_s3{tagx}.parquet"))
    oof = pl.read_parquet(os.path.join(WORK, "models", md, "oof.parquet"), columns=["q", "s", "p2"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    oof = pl.concat([oof, r.select("q", "s").join(oof.select("q", "s"), on=["q", "s"], how="anti")
                    .with_columns(p2=pl.lit(0.0, dtype=oof["p2"].dtype))])
    touched = r.select("q").unique().with_columns(t=pl.lit(True))
    new = (oof.join(r.select("q", "s", "p3"), on=["q", "s"], how="left").join(touched, on="q", how="left")
              .with_columns(p2n=pl.when(pl.col("t").is_null()).then(pl.col("p2")).otherwise(pl.col("p3").fill_null(0.0))))
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]).select(s=id_to_int("entity_id"), h=R.half("entity_id"))
    allS = s1.join(pl.read_parquet(os.path.join(WORK, "pairs", tag, "s1.parquet"), columns=["s"]).with_columns(pl.col("s").cast(pl.Int64)), on="s")
    truth = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    mixed = r.filter(pl.col("grp") == "mixed").select("q").unique().with_columns(mx=pl.lit(True))
    newh = new.join(mixed, on="q", how="left").with_columns(
        p2n=pl.when(pl.col("mx").is_null()).then(pl.col("p2n")).otherwise(pl.col("p2"))).drop("mx")
    ev = allS.filter(pl.col("h") < 500).select("s")
    nh = R._decisions(newh, "p2n", truth.join(ev, on="s"), ev)
    kh = max(nh, key=nh.get)
    log(f"BEST eval, halves protocol (small+bge blend 0.98878): {nh[kh]:.5f} ({kh})")
    na = R._decisions(new, "p2n", truth.join(allS.select("s"), on="s"), allS.select("s"))
    ne = R._decisions(new, "p2n", truth.join(ev, on="s"), ev)
    ka, ke = max(na, key=na.get), max(ne, key=ne.get)
    log(f"BEST all S1 (blend 0.98950): {na[ka]:.5f} ({ka}); eval rule {ke} ({ne[ke]:.5f})")
    json.dump({"note": f"joint stage 3 {fams} (stage3_joint.py)", "x": [], "y": [], "decision": {"default": json.loads(ke)},
               "base": 0, "best": ne[ke]}, open(os.path.join(R.OUT, f"rule{tagx}.json"), "w"), indent=1)
    if test_scores is None:
        return
    m = xgb.train(params, xgb.DMatrix(r.select(feats).to_numpy(), r["label"].to_numpy()), 300)
    t = None
    for k in range(3):     # as rerank.stage3 with fold sides: each fold's test scores through the model, then the mean
        tk, _ = feats_for(load("test", fams, k), fams)
        tk = tk.select("q", "s", **{f"p3_{k}": pl.Series(m.predict(xgb.DMatrix(tk.select(feats).to_numpy())), dtype=pl.Float32)})
        t = tk if t is None else t.join(tk, on=["q", "s"])
    t = t.select("q", "s", p3=pl.mean_horizontal(["p3_0", "p3_1", "p3_2"]))
    te = pl.read_parquet(test_scores).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    xr = pl.read_parquet(os.path.join(R.OUT, "test_rows.parquet"), columns=["q", "s", "p1"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(te.select("q", "s"), on=["q", "s"], how="anti")
    te = pl.concat([te, xr.with_columns(pl.col("p1").cast(te["p1"].dtype), p2=pl.lit(0.0, dtype=te["p2"].dtype)).select(te.columns)])
    tt = t.select("q").unique().with_columns(tch=pl.lit(True))
    te = (te.join(t, on=["q", "s"], how="left").join(tt, on="q", how="left")
            .with_columns(p2=pl.when(pl.col("tch").is_null()).then(pl.col("p2")).otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))
            .select("q", "s", "p1", "p2"))
    dst = os.path.join(R.OUT, f"test_scores_ce{tagx}.parquet")
    te.write_parquet(dst)
    log(f"wrote {dst} ({te.height} rows; {tt.height} records re-scored)")


if __name__ == "__main__":
    main("_" + sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else None)
