"""Name-search df cap at full density: union recall, candidates and search time for max_df 4000 (production),
10000 and 20000 (name_search_diag.py: at 4000 a median name keeps 4 3-grams and 3.2% keep none).
Same queries as dense_blocking_exp.py / blocking_variants.py.

  python name_cap_exp.py US 100000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import time
import numpy as np
from common import log
from candidates import _vec, _wvec, _pad, _comb, _topn, K_NAME, K_NAME_NOADDR, K_ADDR, K_COMB, MAX_DF_WORD
from dense_blocking_exp import load, keys


def main(country="US", n=100_000):
    s1, q = load(country, n)
    tr = q["true_row"].to_numpy()
    truek = keys(np.arange(q.height), tr)
    noaddr = np.flatnonzero(q["addr_missing"].to_numpy())
    va, vc = _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
    Ba, Bc = va.fit_transform(s1["addr"].to_list()), vc.fit_transform(_comb(s1))
    other = [keys(*_topn(va.transform(q["addr"].to_list()), Ba, K_ADDR)),
             keys(*_topn(vc.transform(_comb(q)), Bc, K_COMB))]
    del Ba, Bc
    for cap in (4000, 10000, 20000):
        vn = _vec(cap)
        Bn = vn.fit_transform(_pad(s1["name_ns"]))
        An = vn.transform(_pad(q["name_ns"]))
        t0 = time.time()
        r1, c1 = _topn(An, Bn, K_NAME)
        rr, cc = _topn(An[noaddr], Bn, K_NAME_NOADDR)
        secs = time.time() - t0
        name = np.concatenate([keys(r1, c1), keys(noaddr[rr], cc)])
        U = np.unique(np.concatenate([name] + other))
        log(f"{country} name max_df={cap:5d}: name search recall {np.isin(truek, name).mean():.4f}, "
            f"union recall {np.isin(truek, U).mean():.4f}, {len(U) / q.height:.1f} cand/record, "
            f"name search {secs:.0f}s per {q.height} queries")
        del Bn, An


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "US", int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
