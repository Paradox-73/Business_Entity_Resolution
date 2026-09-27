"""Test candidate pairs with a wider combined search (name + address words): max_df 20000 and top 40 instead of
5000 / 20. Everything else (name, address, e5 searches; all features incl. cos_comb_w with max_df 5000) is production.

Why (27 Sep, measured on labelled train samples, tmp/verify-recall): the shortlist misses 1.49% of true pairs; most
with-address misses sit at busy addresses where the top-20 combined search runs out of slots or drops common words.
This search finds 45% of them and stage 1 ranks 94% of those first. Estimated held-out +0.0011..0.0015 (stage 2,
transformers and stage 3 not rerun in that estimate; stage 2 now sees ~50 candidates per record instead of ~31).

  python blocking_b2.py build [US,India]     -> WORK/pairs/test_b2 (France chunks hard-linked from pairs/test)
  python blocking_b2.py time US 50000        -> time one chunk of 50k US test records (production vs wide)
  python blocking_b2.py predict              -> stage 1 (+ ranks 3-5) and stage 2 of full_cons on pairs/test_b2:
                                               WORK/test_scores_full_cons_test_b2.parquet, WORK/test_rank3_5_test_b2.parquet
  python blocking_b2.py rows                 -> WORK/ce_b2: test close calls + ranks 3-5 (train rows = ce_x), old
                                               transformer scores copied for the families not rescored here
"""
import glob
import json
import os
import shutil
import sys
import time
import numpy as np
import polars as pl
import candidates as C
import pipeline as P
from common import WORK, log

K_COMB2, MAX_DF_COMB2 = 40, 20000
TAG = "test_b2"


class WideIndex(C.CountryIndex):
    def __init__(self, s1, split):
        super().__init__(s1, split)
        self.vcs = C._wvec(MAX_DF_COMB2)
        self.Bcs = self.vcs.fit_transform(C._comb(self.s1))
        self.kc = K_COMB2
        log(f"  wide combined search: max_df {MAX_DF_COMB2}, top {K_COMB2}, vocab {self.Bcs.shape[1]}")


def build(countries):
    out = P.pairs_dir(TAG)
    os.makedirs(out, exist_ok=True)
    for c in ("US", "India", "France"):
        if c in countries:
            continue
        for f in glob.glob(os.path.join(P.pairs_dir("test"), f"{c}_*.parquet")) + [os.path.join(P.pairs_dir("test"), f"{c}.done")]:
            dst = os.path.join(out, os.path.basename(f))
            if not os.path.exists(dst):
                os.link(f, dst)                  # same bytes as production, no extra disk
    C.CountryIndex = WideIndex
    # ~50 candidates per record instead of ~31: smaller chunks than production keep RAM down. India was started with
    # 150k (chunk files are numbered, so a resumed country must keep its size); US uses 100k.
    sizes = {"India": 150_000, "US": 100_000}
    def country_pairs(s1, q, split, on_chunk, skip=None):
        C.CHUNK = sizes[s1["country"][0]]
        return C.country_pairs(s1, q, split, on_chunk, skip)
    P.country_pairs = country_pairs
    P.build("test", TAG)


def time_chunk(country, n):
    s1 = P.load("test", 1).filter(P.pl.col("country") == country)
    q = P.pl.concat([P.load("test", k) for k in (2, 3)]).filter(P.pl.col("country") == country).head(n)
    for cls in (C.CountryIndex, WideIndex):
        t = time.time()
        idx = cls(s1, "test")
        t1 = time.time()
        p = idx.chunk_pairs(q)
        log(f"{cls.__name__}: index {t1 - t:.0f}s, chunk of {n} records {time.time() - t1:.0f}s, "
            f"{p.height / q.height:.1f} candidates per record")


SCORES = os.path.join(WORK, "test_scores_full_cons_test_b2.parquet")
X35 = os.path.join(WORK, "test_rank3_5_test_b2.parquet")
CE = os.path.join(WORK, "ce_b2")


def predict(md_name="full_cons"):
    """pipeline.predict for pairs/test_b2 without the candidate-list output (RAM), with stage-1 ranks 3-5 (topk5.py)
    taken in the same pass. France chunks are production's, so France p1/p2 must equal test_scores_full_cons."""
    md = os.path.join(WORK, "models", md_name)
    res = json.load(open(os.path.join(md, "result.json")))
    d1 = res.get("stage1_dir", md)
    m1 = [P.load_model(os.path.join(d1, "s1_full"))] if os.path.exists(os.path.join(d1, "s1_full.txt")) else         [P.load_model(os.path.join(d1, f"s1_f{i}")) for i in range(P.N_FOLDS)]
    m2 = [P.load_model(os.path.join(md, f"s2_f{i}")) for i in range(P.N_FOLDS)]
    cols1, cols2 = res.get("cols1", C.FEATURES), res.get("cols2", C.FEATURES + P.S2FEATS)
    d = P.pairs_dir(TAG)
    qinfo = pl.read_parquet(os.path.join(d, "q.parquet")).drop("country")
    base, x35 = [], []
    for f in sorted(glob.glob(os.path.join(d, "*_*.parquet"))):
        p = pl.read_parquet(f)
        p = p.with_columns(p1=pl.Series(np.mean([P.score(m, P.X(p, cols1)) for m in m1], axis=0), dtype=pl.Float32))
        base.append(P.topk(p))
        r = p.select("q", "s", "p1").with_columns(r=pl.col("p1").rank("ordinal", descending=True).over("q"))
        x35.append(r.filter((pl.col("r") >= 3) & (pl.col("r") <= 5) & (pl.col("p1") >= 0.005)))
        log(f"  {os.path.basename(f)}: {p.height} pairs, {p['q'].n_unique()} records")
        del p, r
    pl.concat(x35).write_parquet(X35)
    B = P.stage2_features(pl.concat(base), qinfo)
    del base, x35
    qattr = P.load_qattr("test")
    B = P.sibling_features(B, qattr)
    B = P.consensus_features(B, qattr)
    del qattr
    B = B.with_columns(p2=pl.Series(np.mean([P.score(m, P.X(B, cols2)) for m in m2], axis=0), dtype=pl.Float32))
    B.select("q", "s", "p1", "p2").write_parquet(SCORES)
    log(f"wrote {SCORES}: {B.height} rows, {B['q'].n_unique()} records")
    old = pl.read_parquet(os.path.join(WORK, f"test_scores_{md_name}.parquet"))
    s1c = pl.read_parquet(os.path.join(d, "s1.parquet"), columns=["s", "country"])
    j = B.select("q", "s", "p2").join(old.select("q", "s", p2o="p2"), on=["q", "s"], how="full", coalesce=True).join(s1c, on="s")
    for c in ("France", "US", "India"):
        x = j.filter(pl.col("country") == c)
        both = x.filter(pl.col("p2").is_not_null() & pl.col("p2o").is_not_null())
        log(f"  check {c}: pairs new {x['p2'].is_not_null().sum()}, old {x['p2o'].is_not_null().sum()}, both {both.height}, "
            f"max |p2 - old| {(both['p2'] - both['p2o']).abs().max():.6f}")


def rows(fams_rescored=("small", "bge"), fams_copied=()):
    """WORK/ce_b2 for the rerank.py steps: select.json and train rows/scores as in ce_x (train is unchanged), test
    close calls of the new scores + their stage-1 ranks 3-5 (as rerank.extras). For the families not rescored here
    the old test scores are kept only for records whose every new row has one (stage-3 features are per record)."""
    X = os.path.join(WORK, "ce_x")
    os.makedirs(CE, exist_ok=True)
    shutil.copy(os.path.join(X, "select.json"), CE)
    for f in ["train_rows.parquet"] + [f"train_ce_{fam}f{k}.parquet" for fam in fams_rescored + fams_copied for k in range(3)]:
        if os.path.exists(os.path.join(X, f)) and not os.path.exists(os.path.join(CE, f)):
            os.link(os.path.join(X, f), os.path.join(CE, f))
    import rerank
    te = pl.read_parquet(SCORES)
    cc = rerank.close_calls(te).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    xt = (pl.read_parquet(X35).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
            .join(cc.select("q").unique(), on="q").join(cc.select("q", "s"), on=["q", "s"], how="anti")
            .with_columns(p1=pl.col("p1").cast(cc["p1"].dtype), p2=pl.lit(None, dtype=cc["p2"].dtype)).select(cc.columns))
    tr = pl.concat([cc, xt])
    tr.write_parquet(os.path.join(CE, "test_rows.parquet"))
    old = pl.read_parquet(os.path.join(X, "test_rows.parquet"), columns=["q", "s"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    new = tr.select("q", "s").join(old, on=["q", "s"], how="anti")
    log(f"test rows: {cc.height} close calls + {xt.height} ranks 3-5 = {tr.height} ({tr['q'].n_unique()} records); "
        f"pairs not scored before: {new.height} in {new['q'].n_unique()} records")
    for fam in fams_copied:
        for k in range(3):
            src = os.path.join(X, f"test_ce_{fam}f{k}.parquet")
            if not os.path.exists(src):
                continue
            o = pl.read_parquet(src, columns=["q", "s", "ce"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
            x = tr.join(o, on=["q", "s"], how="left")
            full = x.group_by("q").agg(ok=pl.col("ce").is_not_null().all()).filter("ok").select("q")
            x = x.join(full, on="q")
            x.write_parquet(os.path.join(CE, f"test_ce_{fam}f{k}.parquet"))
            log(f"  {fam}f{k}: old scores kept for {full.height} of {tr['q'].n_unique()} records ({x.height} rows)")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2].split(",") if len(sys.argv) > 2 else ["US", "India"])
    elif sys.argv[1] == "time":
        time_chunk(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "predict":
        predict()
    elif sys.argv[1] == "rows":
        rows()
