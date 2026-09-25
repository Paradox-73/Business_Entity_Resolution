"""Score models trained on FULL out-of-fold on another labelled split (tl2 = test-like train split built with the
real search), to check whether that split predicts the leaderboard.

  python evaluate.py <tag> <model_dir> [<model_dir> ...]     e.g. evaluate.py tl2 full full_sib full_cons tlike_xgb_sib

Folds are the same as FULL (crc32(s1_id) % 3; unmatched records by id), so for rows of fold f the fold-f models
(trained on the other folds) are used: every score is out-of-fold. Stage-1 scores are cached per stage-1 model in
WORK/eval_<tag>_<stage1 dir>.parquet, so models sharing a stage 1 (v2, v3, v6) cost one stage-1 pass.
Prints macro F0.5 on the eval half of the split's S1 rows, overall and per country, with each model's own rule.
"""
import glob
import json
import os
import sys
import numpy as np
import polars as pl
from common import WORK, log, macro_f05_df
from pipeline import (pairs_dir, train_meta, attach, topk, stage2_features, sibling_features, consensus_features,
                      load_qattr, load_model, score, X, apply_decision, FEATURES, S2FEATS, SIBFEATS, CONSFEATS, N_FOLDS)


def stage1_base(tag, d1, qmap, cols1):
    cache = os.path.join(WORK, f"eval_{tag}_{os.path.basename(d1)}.parquet")
    smoke = os.environ.get("BER_EVAL_SMOKE")          # smoke test: first chunk only, nothing cached
    if os.path.exists(cache) and not smoke:
        return pl.read_parquet(cache)
    m1 = [load_model(os.path.join(d1, f"s1_f{i}")) for i in range(N_FOLDS)]
    base = []
    for fpath in sorted(glob.glob(os.path.join(pairs_dir(tag), "*_*.parquet")))[:1 if smoke else None]:
        p = attach(pl.read_parquet(fpath), qmap)
        p1 = np.zeros(p.height, np.float32)
        fold = p["fold"].to_numpy()
        for f in range(N_FOLDS):
            m = fold == f
            if m.any():
                p1[m] = score(m1[f], X(p.filter(pl.Series(m)), cols1))
        base.append(topk(p.with_columns(p1=pl.Series(p1))))
        log(f"  stage 1 {os.path.basename(fpath)}")
    B = pl.concat(base)
    if not smoke:
        B.write_parquet(cache)
    return B


def main(tag, dirs):
    s1, t, qmap = train_meta(tag)
    qinfo = pl.read_parquet(os.path.join(pairs_dir(tag), "q.parquet")).drop("country")
    s1e = s1.filter("is_eval").select("s", "country")
    te = t.join(s1e.select("s"), on="s")
    qattr = None
    for name in dirs:
        md = os.path.join(WORK, "models", name)
        res = json.load(open(os.path.join(md, "result.json")))
        d1 = res.get("stage1_dir", md)
        B = stage2_features(stage1_base(tag, d1, qmap, res.get("cols1", FEATURES)), qinfo)
        cols2 = res.get("cols2", FEATURES + S2FEATS)
        if any(c in cols2 for c in SIBFEATS):
            qattr = qattr if qattr is not None else load_qattr(tag)
            B = sibling_features(B, qattr)
            if any(c in cols2 for c in CONSFEATS):
                B = consensus_features(B, qattr)
        ext = ".json" if os.path.exists(os.path.join(md, "s2_f0.json")) else ".txt"
        m2 = [load_model(os.path.join(md, f"s2_f{i}")) for i in range(N_FOLDS)]
        p2 = np.zeros(B.height, np.float32)
        fold = B["fold"].to_numpy()
        for f in range(N_FOLDS):
            m = fold == f
            p2[m] = score(m2[f], X(B.filter(pl.Series(m)), cols2))
        B = B.with_columns(p2=pl.Series(p2))
        prob = "p2" if res.get("use_stage2", True) else "p1"
        M = apply_decision(B, prob, res["decision"], qinfo)
        out = {"all": macro_f05_df(M, te, s1e.select("s"))}
        for c in s1e["country"].unique().sort().to_list():
            sc = s1e.filter(pl.col("country") == c).select("s")
            out[c] = macro_f05_df(M, t.join(sc, on="s"), sc)
        log(f"{tag} | {name} (stage 1 {os.path.basename(d1)}, stage-2 models {ext}, rule {res['decision']}): "
            + ", ".join(f"{k} {v:.5f}" for k, v in out.items()))
        if not os.environ.get("BER_EVAL_SMOKE"):
            json.dump(out, open(os.path.join(md, f"eval_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
