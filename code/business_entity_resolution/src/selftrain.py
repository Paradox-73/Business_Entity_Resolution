"""Self-training on TEST (allowed by the organisers, forum answer 25 Sep): the model's own confident test
decisions are added as extra stage-2 training rows, so stage 2 also learns from test's conditions (US density,
France), then stage 2 is retrained. Stage 1 is unchanged (cached scores are reused).

  python selftrain.py <tag> <base_model_dir> <test_tag>:<countries> [...]
  e.g. python selftrain.py tlike tlike_xgb_cons test:US,India testfr:France
  then: pipeline.py predict <base_model_dir>_self test / testfr

Pseudo-labels from the base model's p2: a row is positive if it is its S2/S3 row's best candidate with
p2 >= POS and the 2nd candidate has p2 <= NEG_2ND; negative if p2 <= NEG; every other row is left out.
They get weight W (true training labels weight 1).

Validation simulates the test situation: the fold-f model also gets fold f's OWN pseudo-labels (from the base
model's out-of-fold p2) and is then scored on fold f's true labels. If self-training only reinforces its own
mistakes, this score drops below the base model's.
"""
import json
import os
import sys
import numpy as np
import polars as pl
import xgboost as xgb
from common import WORK, log
from pipeline import (pairs_dir, train_meta, stage2_features, sibling_features, consensus_features, load_qattr,
                      load_model, score, X, tune, tune_decision, SIBFEATS, CONSFEATS, XGB_PARAMS, XGB_ROUNDS,
                      N_FOLDS, STAGE2_RATE)

POS, NEG_2ND, NEG = 0.97, 0.03, 0.02
W = 0.5
# max pseudo-labelled S2/S3 rows per country: keeps each fold's matrix ~4.5M rows, which fits the 4 GB GPU
# (5M x 57 fitted in v4; 11.5M x 57 ran out of memory)
CAP = {"US": 250_000, "India": 250_000, "France": 450_000}
OWN_RATE = 0.4      # share of a fold's own pseudo-labelled rows that are added to its training set


def pseudo(b, prob):
    r = b.with_columns(rk=pl.col(prob).rank("ordinal", descending=True).over("q"))
    r = r.join(r.filter(pl.col("rk") == 2).select("q", p2nd=prob), on="q", how="left").with_columns(
        pl.col("p2nd").fill_null(0.0))
    pos = (pl.col("rk") == 1) & (pl.col(prob) >= POS) & (pl.col("p2nd") <= NEG_2ND)
    r = r.with_columns(plabel=pl.when(pos).then(1).when(pl.col(prob) <= NEG).then(0).otherwise(None))
    return r.filter(pl.col("plabel").is_not_null()).drop("rk", "p2nd")


def add_group_feats(b, split, cols2):
    if any(c in cols2 for c in SIBFEATS):
        qattr = load_qattr(split)
        b = sibling_features(b, qattr)
        if any(c in cols2 for c in CONSFEATS):
            b = consensus_features(b, qattr)
        del qattr
    return b


def test_pseudo(spec, cols2, s1tag, m2):
    ttag, countries = spec.split(":")
    countries = countries.split(",")
    qinfo = pl.read_parquet(os.path.join(pairs_dir(ttag), "q.parquet"))
    keep = qinfo.filter(pl.col("country").is_in(countries))
    B = pl.read_parquet(os.path.join(WORK, f"test_base_{ttag}_{s1tag}.parquet")).join(keep.select("q"), on="q")
    B = stage2_features(B, qinfo.drop("country"))
    split = ttag if os.path.exists(os.path.join(WORK, f"{ttag}_s2.parquet")) else "test"
    B = add_group_feats(B, split, cols2)
    B = B.with_columns(pb=pl.Series(np.mean([score(m, X(B, cols2)) for m in m2], axis=0), dtype=pl.Float32))
    P = pseudo(B, "pb").join(keep.select("q", "country"), on="q")
    del B
    parts = []
    for c in countries:
        pc = P.filter(pl.col("country") == c)
        n = pc["q"].n_unique()
        rate = min(1.0, CAP.get(c, 700_000) / max(n, 1))
        pc = pc.filter((pl.col("q").hash(seed=13) % 100000) < rate * 100000)
        log(f"test {ttag} {c}: pseudo rows {pc.height} ({pc['q'].n_unique()} S2/S3 rows of {n}), positives {int(pc['plabel'].sum())}")
        parts.append(pc)
    P = pl.concat(parts)
    return X(P, cols2), P["plabel"].to_numpy().astype(np.float32)


def main(tag, base, specs):
    md0 = os.path.join(WORK, "models", base)
    res = json.load(open(os.path.join(md0, "result.json")))
    cols2 = res["cols2"]
    md1 = res["stage1_dir"]
    s1tag = os.path.basename(md1)
    m2 = [load_model(os.path.join(md0, f"s2_f{i}")) for i in range(N_FOLDS)]
    Xt, yt = [], []
    for spec in specs:
        a, b = test_pseudo(spec, cols2, s1tag, m2)
        Xt.append(a)
        yt.append(b)
    Xt, yt = np.vstack(Xt), np.concatenate(yt)
    log(f"test pseudo rows total {len(yt)}, positives {int(yt.sum())}")

    s1, t, _ = train_meta(tag)
    qinfo = pl.read_parquet(os.path.join(pairs_dir(tag), "q.parquet")).drop("country")
    s1e = s1.filter("is_eval").select("s")
    te = t.join(s1e, on="s")
    B = pl.read_parquet(os.path.join(md1, "stage1_base.parquet"))
    B = add_group_feats(stage2_features(B, qinfo), "train", cols2)
    B = B.join(pl.read_parquet(os.path.join(md0, "oof.parquet"), columns=["q", "s", "p2"]).rename({"p2": "pb"}),
               on=["q", "s"], how="left")
    rate = STAGE2_RATE.get(tag, 1.0)
    p2 = np.zeros(B.height, np.float32)
    models = []
    for f in range(N_FOLDS):
        samp = (pl.col("q").hash(seed=5) % 1000) < rate * 1000
        tr = B.filter((pl.col("fold") != f) & pl.col("eligible") & samp)
        own = pseudo(B.filter((pl.col("fold") == f) & samp &
                              ((pl.col("q").hash(seed=17) % 1000) < OWN_RATE * 1000)), "pb")
        Xf = np.vstack([X(tr, cols2), X(own, cols2), Xt])
        yf = np.concatenate([tr["label"].to_numpy().astype(np.float32), own["plabel"].to_numpy().astype(np.float32), yt])
        wf = np.concatenate([np.ones(tr.height, np.float32), np.full(own.height + len(yt), W, np.float32)])
        m = xgb.train(XGB_PARAMS[2], xgb.QuantileDMatrix(Xf, yf, weight=wf), XGB_ROUNDS[2])
        del Xf, yf, wf
        models.append(m)
        msk = (B["fold"] == f).to_numpy()
        p2[msk] = score(m, X(B.filter(pl.Series(msk)), cols2))
        log(f"fold {f}: {tr.height} labelled + {own.height} own-fold pseudo + {len(yt)} test pseudo rows")
    B = B.with_columns(p2=pl.Series(p2))
    th2, sc2, (g2, gs2) = tune(B, "p2", te, s1e, qinfo)
    dec, sc_dec = tune_decision(B, "p2", te, s1e, g2, gs2)
    log(f"SELF-TRAINED stage 2: thr {g2} -> {gs2:.5f}; decision {dec} -> {sc_dec:.5f}; "
        f"base model {base}: {res['decision_score']:.5f}")
    md = md0 + "_self"
    os.makedirs(md, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(os.path.join(md, f"s2_f{i}.json"))
    res.update({"stage2_global": [g2, gs2], "stage2_seg": sc2, "thresholds": th2, "decision": dec,
                "decision_score": sc_dec, "self_training": {"base": base, "specs": specs, "POS": POS,
                "NEG_2ND": NEG_2ND, "NEG": NEG, "W": W, "test_pseudo_rows": int(len(yt))}})
    json.dump(res, open(os.path.join(md, "result.json"), "w"), indent=1, default=float)
    B.select("q", "s", "p1", "p2", "label", "fold").write_parquet(os.path.join(md, "oof.parquet"))
    log("saved", md)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
