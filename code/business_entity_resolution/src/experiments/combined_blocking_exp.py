"""All blocking improvements together, added one at a time (cumulative), to see how much they overlap:
  production -> name max_df 20000 -> word searches min_df=1 -> no-address name k=100 -> + dense e5 top-10.
Same queries as dense_blocking_exp.py (Latin names, true S1 in the eval half, full-density S1 index).

  python combined_blocking_exp.py US 100000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import numpy as np
import torch
from common import log
from candidates import _vec, _wvec, _pad, _comb, _topn, K_NAME, K_NAME_NOADDR, K_ADDR, K_COMB, MAX_DF_NAME, MAX_DF_WORD
from dense_blocking_exp import load, keys, dense_topk, MODELS
from blocking_variants import hashed
from embed import Encoder


def main(country="US", n=100_000):
    s1, q = load(country, n)
    tr = q["true_row"].to_numpy()
    truek = keys(np.arange(q.height), tr)
    noaddr = np.flatnonzero(q["addr_missing"].to_numpy())

    def name_search(cap, k_noaddr):
        v = _vec(cap)
        B = v.fit_transform(_pad(s1["name_ns"]))
        A = v.transform(_pad(q["name_ns"]))
        rr, cc = _topn(A[noaddr], B, k_noaddr)
        return np.concatenate([keys(*_topn(A, B, K_NAME)), keys(noaddr[rr], cc)])

    def word_searches(min_df):
        if min_df == 2:
            va, vc = _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
            Ba, Bc = va.fit_transform(s1["addr"].to_list()), vc.fit_transform(_comb(s1))
            return [keys(*_topn(va.transform(q["addr"].to_list()), Ba, K_ADDR)),
                    keys(*_topn(vc.transform(_comb(q)), Bc, K_COMB))]
        Qa, Sa = hashed(s1["addr"].to_list(), q["addr"].to_list(), min_df=1, max_df=MAX_DF_WORD)
        Qc, Sc = hashed(_comb(s1), _comb(q), min_df=1, max_df=MAX_DF_WORD)
        return [keys(*_topn(Qa, Sa, K_ADDR)), keys(*_topn(Qc, Sc, K_COMB))]

    def report(tag, parts):
        U = np.unique(np.concatenate(parts))
        log(f"{country} {tag:34s} union recall {np.isin(truek, U).mean():.4f}  {len(U) / q.height:5.1f} cand/record")

    w2 = word_searches(2)
    report("production", [name_search(MAX_DF_NAME, K_NAME_NOADDR)] + w2)
    n20 = name_search(20000, K_NAME_NOADDR)
    report("+ name max_df 20000", [n20] + w2)
    w1 = word_searches(1)
    report("+ word min_df=1", [n20] + w1)
    n20k = name_search(20000, 100)
    report("+ no-address name k=100", [n20k] + w1)
    enc = Encoder(MODELS[0])
    _, _, top = dense_topk(enc, s1, q, 10)
    del enc
    torch.cuda.empty_cache()
    dense = keys(np.repeat(np.arange(q.height), 10), top.ravel())
    report("+ dense e5 top-10 (all of the above)", [n20k] + w1 + [dense])
    report("production + dense e5 top-10 only", [name_search(MAX_DF_NAME, K_NAME_NOADDR)] + w2 + [dense])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "US", int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
