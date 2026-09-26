"""Stage-1 candidates ranked 3..5 per record (stage 2 and the reranker only see the top 2).

On held-out, 0.65% of true pairs are shortlisted but ranked 3rd or lower by stage 1, so no later step can pick them.
This writes those lower-ranked candidates (p1 >= MINP) so the transformer can rescore them for close calls.

  python topk5.py train [model_dir] -> WORK/models/<model_dir>/stage1_rank3_5.parquet  (out-of-fold p1, fold models)
  python topk5.py test  [model_dir] -> WORK/test_rank3_5_test_<model_dir>.parquet     (same stage-1 model as test)
model_dir defaults to 'full' (stage 1 of v3 and v6). Columns: q, s, p1, r (rank by p1 in the record), [label, fold].
Resumable: one part file per chunk under WORK/rank3_5_<split>_<model_dir>/.
"""
import glob
import json
import os
import sys
import numpy as np
import polars as pl
from common import WORK, log
from pipeline import pairs_dir, train_meta, attach, load_model, score, X, FEATURES, N_FOLDS

KMAX, MINP = 5, 0.005


def ranks(p):
    r = p.with_columns(r=pl.col("p1").rank("ordinal", descending=True).over("q"))
    return r.filter((pl.col("r") >= 3) & (pl.col("r") <= KMAX) & (pl.col("p1") >= MINP))


def main(split, md_name="full"):
    md = os.path.join(WORK, "models", md_name)
    res = json.load(open(os.path.join(md, "result.json")))
    cols1 = res.get("cols1", FEATURES)
    tag = "full" if split == "train" else "test"
    pdir = os.path.join(WORK, f"rank3_5_{split}_{md_name}")
    os.makedirs(pdir, exist_ok=True)
    if split == "train":
        _, _, qmap = train_meta(tag)
        models = [load_model(os.path.join(md, f"s1_f{i}")) for i in range(N_FOLDS)]
    else:
        full = os.path.join(md, "s1_full")
        models = [load_model(full)] if (os.path.exists(full + ".txt") or os.path.exists(full + ".json")) else \
            [load_model(os.path.join(md, f"s1_f{i}")) for i in range(N_FOLDS)]
    files = sorted(glob.glob(os.path.join(pairs_dir(tag), "*_*.parquet")))
    if os.environ.get("BER_TOPK_SMOKE"):
        files = files[:1]
    parts = []
    for fpath in files:
        dst = os.path.join(pdir, os.path.basename(fpath))
        if not os.path.exists(dst):
            p = pl.read_parquet(fpath)
            if split == "train":
                p = attach(p, qmap)
                p1 = np.zeros(p.height, np.float32)
                fold = p["fold"].to_numpy()
                for f in range(N_FOLDS):
                    m = fold == f
                    if m.any():
                        p1[m] = score(models[f], X(p.filter(pl.Series(m)), cols1))
                keep = ["q", "s", "p1", "r", "label", "fold"]
            else:
                p1 = np.mean([score(m, X(p, cols1)) for m in models], axis=0)
                keep = ["q", "s", "p1", "r"]
            out = ranks(p.select([c for c in ("q", "s", "label", "fold") if c in p.columns]).with_columns(
                p1=pl.Series(p1, dtype=pl.Float32))).select(keep)
            out.write_parquet(dst + ".tmp")
            os.replace(dst + ".tmp", dst)
            log(f"  {os.path.basename(fpath)}: {p.height} pairs -> {out.height} rank 3-{KMAX} candidates")
            del p
        parts.append(pl.read_parquet(dst))
    B = pl.concat(parts)
    out = os.path.join(md, "stage1_rank3_5.parquet") if split == "train" else os.path.join(WORK, f"test_rank3_5_test_{md_name}.parquet")
    if os.environ.get("BER_TOPK_SMOKE"):
        out = out.replace(".parquet", "_smoke.parquet")
    B.write_parquet(out)
    msg = f"; positives {int(B['label'].sum())}" if "label" in B.columns else ""
    log(f"wrote {out}: {B.height} rows, {B['q'].n_unique()} records{msg}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
