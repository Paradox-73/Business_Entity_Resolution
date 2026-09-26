"""Test candidate pairs with a wider combined search (name + address words): max_df 20000 and top 40 instead of
5000 / 20. Everything else (name, address, e5 searches; all features incl. cos_comb_w with max_df 5000) is production.

Why (27 Sep, measured on labelled train samples, tmp/verify-recall): the shortlist misses 1.49% of true pairs; most
with-address misses sit at busy addresses where the top-20 combined search runs out of slots or drops common words.
This search finds 45% of them and stage 1 ranks 94% of those first. Estimated held-out +0.0011..0.0015 (stage 2,
transformers and stage 3 not rerun in that estimate; stage 2 now sees ~50 candidates per record instead of ~31).

  python blocking_b2.py build [US,India]     -> WORK/pairs/test_b2 (France chunks hard-linked from pairs/test)
  python blocking_b2.py time US 50000        -> time one chunk of 50k US test records (production vs wide)
"""
import glob
import os
import sys
import time
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
    C.CHUNK = 150_000       # ~50 candidates per record instead of ~31: smaller chunks keep RAM as in production
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


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2].split(",") if len(sys.argv) > 2 else ["US", "India"])
    elif sys.argv[1] == "time":
        time_chunk(sys.argv[2], int(sys.argv[3]))
