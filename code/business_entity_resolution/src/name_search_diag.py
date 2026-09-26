"""Why does the name search miss records whose cleaned name is IDENTICAL to their S1's (e.g. web domains)?

For US train: fit the production name vectoriser (char 3-grams of name_ns, min_df=2, max_df=4000) on all US S1,
then for records whose name_ns equals their true S1's name_ns: how many 3-grams survive the df filter, the cosine
and rank of the true S1, and recall of the name search. Repeats with other max_df caps to see the trade-off.

  python name_search_diag.py
"""
import os
import time
import numpy as np
import polars as pl
from common import WORK, log, read_truth
from candidates import _pad, _topn, K_NAME
from sklearn.feature_extraction.text import TfidfVectorizer


def main(country="US", n=30_000):
    s1 = (pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "name_ns", "country"])
            .filter(pl.col("country") == country).with_row_index("s_row"))
    t = read_truth()
    q = pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet"))
                   .filter((pl.col("country") == country) & ~pl.col("name_nonlatin"))
                   .select("entity_id", "name_ns", "name_is_domain").collect() for k in (2, 3)])
    q = (q.join(t.rename({"q_id": "entity_id"}), on="entity_id")
          .join(s1.select(s1_id="entity_id", true_row="s_row", true_ns="name_ns"), on="s1_id"))
    same = q.filter(pl.col("name_ns") == pl.col("true_ns"))
    dup = s1.group_by("name_ns").len()
    same = same.join(dup.rename({"len": "n_same_name"}), on="name_ns").filter(pl.col("n_same_name") == 1)
    same = same.sample(min(n, same.height), seed=0)
    log(f"{country}: records whose cleaned name equals their S1's AND the name is unique in S1: {same.height}"
        f" (web domains {same['name_is_domain'].mean():.3f})")
    tr = same["true_row"].to_numpy()
    for min_df, max_df in ((2, 4000), (1, 4000), (2, 20000), (1, 20000)):
        t0 = time.time()
        v = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=min_df, max_df=max_df, sublinear_tf=True,
                            dtype=np.float32, lowercase=False)
        B = v.fit_transform(_pad(s1["name_ns"]))
        A = v.transform(_pad(same["name_ns"]))
        nnz = np.diff(A.indptr)
        r, c = _topn(A, B, K_NAME)
        found = np.zeros(len(tr), bool)
        found[r[c == tr[r]]] = True
        log(f"  min_df={min_df} max_df={max_df}: 3-grams kept per name median {np.median(nnz):.0f} "
            f"(0 kept: {np.mean(nnz == 0):.4f}), name search top-{K_NAME} recall {found.mean():.4f}; "
            f"web domains {found[same['name_is_domain'].to_numpy()].mean():.4f}  ({time.time() - t0:.0f}s)")
        if (min_df, max_df) == (2, 4000):
            miss = np.flatnonzero(~found)[:8]
            for i in miss:
                terms = v.inverse_transform(A[i])[0]
                log(f"    miss: {same['name_ns'][int(i)]!r} kept 3-grams {list(terms)}")
        del B, A


if __name__ == "__main__":
    main()
