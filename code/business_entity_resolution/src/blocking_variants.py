"""Blocking and address-cleaning variants against the production shortlist, at full train density.

Same queries as dense_blocking_exp.py (n matched Latin-name S2/S3 records of one country whose true S1 is in the
eval half; index = ALL train S1 rows of the country). Every variant replaces ONE production search (or the address
cleaning) and reports the union recall and candidates per record, so gains and costs are comparable:

  hash_min2     hashing TF-IDF with min_df=2 (sanity check: should reproduce the production word searches)
  min_df1       word searches keep terms found in only ONE S1 row (the most specific ones)
  bm25          BM25 weighting instead of TF-IDF cosine for the word searches
  k sweeps      comb k 20 -> 30/40, addr k 10 -> 20, no-address name k 30 -> 100
  no_abbrev     address cleaning without the learned short-form map (does the map help?)
  state_fix     India only: TN/DL/OD state codes mapped to the names S1 uses (the pooled map sends tn -> tennessee)
Also writes WORK/blocking_misses_<country>.tsv: shortlist misses that have an address and a unique S1 name.

  python blocking_variants.py US 100000
"""
import json
import os
import sys
import time
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from common import WORK, log
from candidates import (_vec, _wvec, _pad, _comb, _topn, K_NAME, K_NAME_NOADDR, K_ADDR, K_COMB,
                        MAX_DF_NAME, MAX_DF_WORD)
from normalize import addr_exprs
from dense_blocking_exp import load, keys

NF = 1 << 24


def hashed(s_text, q_text, min_df, max_df, bm25=False, k1=1.2, b=0.75):
    """Word 1-2-gram search matrices without a vocabulary dict (min_df=1 would make it huge).
    TF-IDF: sublinear tf x smooth idf, L2 rows (as sklearn). BM25: saturated tf x BM25 idf on S1, binary query."""
    hv = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False,
                           n_features=NF, alternate_sign=False, norm=None, dtype=np.float32)
    S = hv.transform(s_text).tocsr()
    Q = hv.transform(q_text).tocsr()
    n = S.shape[0]
    df = np.bincount(S.indices, minlength=NF)
    keep = (df >= min_df) & (df <= max_df)
    if bm25:
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(S.sum(1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        S = S.tocoo()
        S = sp.csr_matrix((idf[S.col] * S.data * (k1 + 1) / (S.data + norm[S.row]), (S.row, S.col)), shape=S.shape)
        Q.data[:] = 1.0
    else:
        idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        for M in (S, Q):
            M.data = (1 + np.log(M.data)) * idf[M.indices]
    colmask = sp.diags(keep.astype(np.float32))
    S, Q = (S @ colmask).tocsr(), (Q @ colmask).tocsr()
    if not bm25:
        for M in (S, Q):
            nr = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
            nr[nr == 0] = 1
            M.data /= np.repeat(nr, np.diff(M.indptr)).astype(np.float32)
    S.eliminate_zeros()
    Q.eliminate_zeros()
    return Q.astype(np.float32), S.astype(np.float32)


def recompute_addr(df, abbrev):
    """Re-clean business_address with another short-form map (script map as in production)."""
    maps = json.load(open(os.path.join(WORK, "maps.json"), encoding="utf8"))
    return df.with_columns(addr_exprs(maps["script"], abbrev)[0])


def main(country="US", n=100_000):
    s1, q = load(country, n)
    raw_cols = ["entity_id", "business_address", "business_name"]
    # row order must survive the joins: s_row / true_row are positions in these frames
    s1 = s1.join(pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=raw_cols).drop("business_name"),
                 on="entity_id", how="left", maintain_order="left")
    q = q.join(pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"), columns=raw_cols[:2])
                          for k in (2, 3)]), on="entity_id", how="left", maintain_order="left")
    assert (s1["s_row"].to_numpy() == np.arange(s1.height)).all()
    tr = q["true_row"].to_numpy()
    truek = keys(np.arange(q.height), tr)
    noaddr = np.flatnonzero(q["addr_missing"].to_numpy())
    log(f"{country}: S1 {s1.height}, queries {q.height}")

    def run(A, B, k, rows=None):
        r, c = _topn(A, B, k)
        if rows is not None:
            r = rows[r]
        return np.unique(keys(r, c))

    # ---- production searches
    t0 = time.time()
    vn, va, vc = _vec(MAX_DF_NAME), _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
    Bn, Ba, Bc = vn.fit_transform(_pad(s1["name_ns"])), va.fit_transform(s1["addr"].to_list()), vc.fit_transform(_comb(s1))
    An, Aa, Ac = vn.transform(_pad(q["name_ns"])), va.transform(q["addr"].to_list()), vc.transform(_comb(q))
    base = {"name": run(An, Bn, K_NAME), "addr": run(Aa, Ba, K_ADDR), "comb": run(Ac, Bc, K_COMB),
            "noaddr": run(An[noaddr], Bn, K_NAME_NOADDR, noaddr)}
    log(f"production searches {time.time() - t0:.0f}s")

    # acronym records: the record's name is the initials of its true S1 core name ("SMT" = "south memorial trust")
    ini = s1["name_core"].str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    acr = ((ini.gather(tr) == q["name_ns"]) & (q["name_ns"].str.len_chars() <= 5)).to_numpy()

    def report(tag, **repl):
        parts = dict(base, **repl)
        U = np.unique(np.concatenate(list(parts.values())))
        hit = np.isin(truek, U)
        log(f"  {tag:28s} union recall {hit.mean():.4f}  cand/record {len(U) / q.height:5.1f}  "
            f"acronym recs {hit[acr].mean() if acr.any() else float('nan'):.4f}  "
            + "  ".join(f"{k} {np.isin(truek, v).mean():.4f}" for k, v in repl.items()))
        return hit

    hit0 = report("PRODUCTION")
    log(f"acronym records in the sample: {acr.sum()}")
    # S1 initials as an extra word in the name+address documents, so "tc | 12 rue x" can meet "tourcoing club | 12 rue x"
    vi = _wvec(MAX_DF_WORD)
    Bi = vi.fit_transform((s1["name_core"] + " " + ini + " " + s1["addr"]).to_list())
    report("comb + S1 initials", comb=run(vi.transform(_comb(q)), Bi, K_COMB))
    del Bi, vi

    # ---- misses worth reading: address present, true S1 name unique
    miss = ~hit0 & ~q["addr_missing"].to_numpy() & (q["true_chain"].to_numpy() == 1)
    sel = np.flatnonzero(miss)
    comb_cos = np.asarray(Ac[sel].multiply(Bc[tr[sel]]).sum(1)).ravel()
    m = q[sel].select("entity_id", q_name="business_name", q_addr="business_address", q_addr_clean="addr").with_columns(
        comb_cos_true=pl.Series(comb_cos))
    m = m.with_columns(s1_name=s1["business_name"].gather(tr[sel]), s1_addr=s1["business_address"].gather(tr[sel]),
                       s1_addr_clean=s1["addr"].gather(tr[sel]))
    m.write_csv(os.path.join(WORK, f"blocking_misses_{country}.tsv"), separator="\t")
    log(f"misses with address and a unique S1 name: {len(sel)} of {(~hit0).sum()} -> blocking_misses_{country}.tsv")

    # ---- k sweeps (cost vs recall)
    for k in (30, 40):
        report(f"comb k={k}", comb=run(Ac, Bc, k))
    report("addr k=20", addr=run(Aa, Ba, 20))
    report("noaddr name k=100", noaddr=run(An[noaddr], Bn, 100, noaddr))
    del An, Bn

    # ---- hashing TF-IDF / min_df / BM25 for the two word searches
    sa, qa, sc, qc = s1["addr"].to_list(), q["addr"].to_list(), _comb(s1), _comb(q)
    for tag, kw in (("hash_min2", dict(min_df=2)), ("min_df1", dict(min_df=1)), ("bm25", dict(min_df=2, bm25=True)),
                    ("bm25 min_df1", dict(min_df=1, bm25=True))):
        t0 = time.time()
        Qa, Sa = hashed(sa, qa, max_df=MAX_DF_WORD, **kw)
        ra = run(Qa, Sa, K_ADDR)
        del Qa, Sa
        Qc, Sc = hashed(sc, qc, max_df=MAX_DF_WORD, **kw)
        rc = run(Qc, Sc, K_COMB)
        del Qc, Sc
        report(f"{tag} ({time.time() - t0:.0f}s)", addr=ra, comb=rc)

    # ---- address cleaning ablations (re-clean both sides, rebuild the two word searches)
    maps = json.load(open(os.path.join(WORK, "maps.json"), encoding="utf8"))
    variants = [("no_abbrev", {})]
    if country == "India":
        fix = dict(maps["abbrev"], tn="tamil nadu", dl="delhi", od="orissa", odisha="orissa")
        variants.append(("state_fix", fix))
    for tag, ab in variants:
        s1v, qv = recompute_addr(s1, ab), recompute_addr(q, ab)
        va2, vc2 = _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
        Ba2, Bc2 = va2.fit_transform(s1v["addr"].to_list()), vc2.fit_transform(_comb(s1v))
        Aa2, Ac2 = va2.transform(qv["addr"].to_list()), vc2.transform(_comb(qv))
        report(tag, addr=run(Aa2, Ba2, K_ADDR), comb=run(Ac2, Bc2, K_COMB))
        if tag == "state_fix":
            # effect on the address FEATURE of true pairs whose record ends in one of the fixed codes
            from rapidfuzz import fuzz
            from rapidfuzz.process import cpdist
            code = q["business_address"].str.split(",").list.last().str.strip_chars().is_in(["TN", "DL", "OD"]).to_numpy()
            idx = np.flatnonzero(code)
            for nm, a, b in (("before", q["addr"], s1["addr"]), ("after", qv["addr"], s1v["addr"])):
                ts = cpdist(a.gather(idx).to_list(), b.gather(tr[idx]).to_list(), scorer=fuzz.token_set_ratio, workers=-1)
                log(f"    a_tset of true pairs with TN/DL/OD records ({len(idx)}): {nm} mean {ts.mean():.1f}, "
                    f">=90 {np.mean(ts >= 90):.3f}")
        del Ba2, Bc2, Aa2, Ac2, s1v, qv


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "US", int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
