"""Ordinal cleaning check: records write 'eleventh street' where S1 writes '11th street' (US: 1.3% of record
addresses vs 0.36% of S1 use ordinal words). Normalise both sides (words -> digits, '11th' -> '11') and rebuild the
two word searches; report union recall and the address token-set of affected true pairs.
Same queries as dense_blocking_exp.py.

  python ordinal_exp.py US 100000
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from common import WORK, log
from candidates import _vec, _wvec, _pad, _comb, _topn, K_NAME, K_NAME_NOADDR, K_ADDR, K_COMB, MAX_DF_NAME, MAX_DF_WORD
from dense_blocking_exp import load, keys

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
       "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
       "twentieth"]
WORDS = r"\b(" + "|".join(ORD) + r")\b"


def ordinal(e):
    e = e.str.replace_many([f" {w} " for w in ORD], [f" {i + 1} " for i in range(len(ORD))])
    return e.str.replace_all(r"\b(\d+)(?:st|nd|rd|th)\b", "$1")


def main(country="US", n=100_000):
    s1, q = load(country, n)
    tr = q["true_row"].to_numpy()
    truek = keys(np.arange(q.height), tr)
    noaddr = np.flatnonzero(q["addr_missing"].to_numpy())
    vn = _vec(MAX_DF_NAME)
    Bn = vn.fit_transform(_pad(s1["name_ns"]))
    An = vn.transform(_pad(q["name_ns"]))
    name = np.concatenate([keys(*_topn(An, Bn, K_NAME)), keys(noaddr[_topn(An[noaddr], Bn, K_NAME_NOADDR)[0]],
                                                              _topn(An[noaddr], Bn, K_NAME_NOADDR)[1])])
    del Bn, An
    affected = (q["addr"].str.contains(WORDS) | s1["addr"].gather(tr).str.contains(WORDS)).to_numpy()
    for tag, fs, fq in (("production", s1, q),
                        ("ordinal fix", s1.with_columns(addr=ordinal(" " + pl.col("addr") + " ").str.strip_chars()),
                         q.with_columns(addr=ordinal(" " + pl.col("addr") + " ").str.strip_chars()))):
        va, vc = _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
        Ba, Bc = va.fit_transform(fs["addr"].to_list()), vc.fit_transform(_comb(fs))
        U = np.unique(np.concatenate([name, keys(*_topn(va.transform(fq["addr"].to_list()), Ba, K_ADDR)),
                                      keys(*_topn(vc.transform(_comb(fq)), Bc, K_COMB))]))
        hit = np.isin(truek, U)
        idx = np.flatnonzero(affected)
        ts = cpdist(fq["addr"].gather(idx).to_list(), fs["addr"].gather(tr[idx]).to_list(),
                    scorer=fuzz.token_set_ratio, workers=-1)
        log(f"{country} {tag:12s}: union recall {hit.mean():.4f} (affected records {hit[affected].mean():.4f}, "
            f"n={affected.sum()}), {len(U) / q.height:.1f} cand/record; a_tset of affected true pairs mean {ts.mean():.1f}, "
            f">=90 {np.mean(ts >= 90):.3f}")
        del Ba, Bc


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "US", int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
