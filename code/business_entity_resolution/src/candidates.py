"""Blocking (shortlist) + pair features, one country at a time, Source 2/3 rows in chunks.

Blocking, searched from each S2/S3 row towards S1 rows of the SAME country (union of):
  - name:      TF-IDF over char 3-grams of the space-free core name, top K_NAME
                (top K_NAME_NOADDR when the S2/S3 row has no address: name is its only evidence)
  - address:   TF-IDF over word 1-2 grams of the cleaned address, top K_ADDR
  - combined:  TF-IDF over word 1-2 grams of core name + address, top K_COMB
  - embedding: for non-Latin-script names only, fine-tuned multilingual e5 (embed.py), top K_EMB
Search matrices drop terms shared by more than MAX_DF_* S1 rows (keeps search cost bounded; word-level
terms stay informative at full density, char 3-grams of addresses do not - EXPERIMENTS.md exp 7);
similarity FEATURES use the full vocabulary.
"""
import os
import time
import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import WORK, log

K_NAME = 10
K_NAME_NOADDR = 30
K_ADDR = 10
K_COMB = 20
K_EMB = 10
MAX_DF_NAME = 4000
MAX_DF_WORD = 5000
CHUNK = 250_000
THREADS = 11
DEV = "cuda" if torch.cuda.is_available() else "cpu"

S_COLS = ["entity_id", "name_full", "name_core", "name_ns", "addr", "addr_nums", "business_address"]
Q_COLS = S_COLS + ["src", "name_is_domain", "name_has_alias", "name_nonlatin", "addr_missing", "addr_nonlatin"]
_ALNUM = r"[a-z0-9]*\d[a-z0-9]*(?:[/\-][a-z0-9]+)*"


def _vec(max_df):
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32, lowercase=False)


def _wvec(max_df):
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32, lowercase=False)


def _comb(df):
    return (df["name_core"] + " " + df["addr"]).to_list()


def _pad(series):
    return (" " + series + " ").to_list()


def _topn(A, B, k):
    C = sp_matmul_topn(A, B, top_n=k, threshold=0.01, n_threads=THREADS).tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64)


def _rowdot(A, B, qi, si):
    out = np.empty(len(qi), np.float32)
    for a in range(0, len(qi), 2_000_000):
        b = a + 2_000_000
        out[a:b] = np.asarray(A[qi[a:b]].multiply(B[si[a:b]]).sum(1)).ravel()
    return out


def _fuzz(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _alnum(col):
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"\s*([/\-])\s*", "$1")
              .str.extract_all(_ALNUM).list.unique())


class CountryIndex:
    """Holds the S1 rows of one country, fitted vectorisers and (optionally) S1 name embeddings."""

    def __init__(self, s1, split):
        self.s1 = (s1.select(S_COLS).with_row_index("s_row")
                     .with_columns(s_name_count=pl.len().over("name_core").cast(pl.Float32),
                                   alnum=_alnum("business_address")))
        self.vn, self.va, self.vc = _vec(MAX_DF_NAME), _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
        self.Bn = self.vn.fit_transform(_pad(self.s1["name_ns"]))
        self.Ba = self.va.fit_transform(self.s1["addr"].to_list())
        self.Bc = self.vc.fit_transform(_comb(self.s1))
        self.fn, self.fa = _vec(1.0), _vec(1.0)
        self.Fn = self.fn.fit_transform(_pad(self.s1["name_ns"]))
        self.Fa = self.fa.fit_transform(_pad(self.s1["addr"]))
        self.E, self.qemb = None, None
        path = os.path.join(WORK, "emb", f"{split}_s1.npz")
        if os.path.exists(path):
            z = np.load(path)
            pos = dict(zip(z["ids"].tolist(), range(len(z["ids"]))))
            rows = [pos.get(e, -1) for e in self.s1["entity_id"].to_list()]
            if max(rows) >= 0:
                E = np.zeros((self.s1.height, z["v"].shape[1]), np.float16)
                ok = np.array(rows) >= 0
                E[ok] = z["v"][np.array(rows)[ok]]
                self.E = torch.tensor(E, device=DEV)
                zq = np.load(os.path.join(WORK, "emb", f"{split}_q.npz"))
                self.qemb = (dict(zip(zq["ids"].tolist(), range(len(zq["ids"])))), zq["v"])
        log(f"  S1 index: {self.s1.height} rows, name vocab {self.Bn.shape[1]}/{self.Fn.shape[1]}, "
            f"addr vocab {self.Ba.shape[1]}/{self.Fa.shape[1]}, embeddings {'yes' if self.E is not None else 'no'}")

    def _emb(self, q):
        """(q_rows, emb matrix) for rows of this chunk that have an embedding."""
        if self.E is None:
            return None, None
        pos, V = self.qemb
        idx = [(i, pos[e]) for i, e in enumerate(q["entity_id"].to_list()) if e in pos]
        if not idx:
            return None, None
        qi, vi = map(np.array, zip(*idx))
        return qi, torch.tensor(V[vi], device=DEV)

    def chunk_pairs(self, q):
        """Candidate pairs + features for a chunk of S2/S3 rows (DataFrame with Q_COLS)."""
        q = q.select(Q_COLS).with_row_index("q_row").with_columns(alnum=_alnum("business_address"))
        t0 = time.time()
        An = self.vn.transform(_pad(q["name_ns"]))
        Aa = self.va.transform(q["addr"].to_list())
        Ac = self.vc.transform(_comb(q))
        r1, c1 = _topn(An, self.Bn, K_NAME)
        r2, c2 = _topn(Aa, self.Ba, K_ADDR)
        r3, c3 = _topn(Ac, self.Bc, K_COMB)
        noaddr = np.flatnonzero(q["addr_missing"].to_numpy())
        if len(noaddr):
            rr, cc = _topn(An[noaddr], self.Bn, K_NAME_NOADDR)
            r1, c1 = np.concatenate([r1, noaddr[rr]]), np.concatenate([c1, cc])
        keys = [r1 * (1 << 32) + c1, r2 * (1 << 32) + c2, r3 * (1 << 32) + c3]
        qe_rows, QE = self._emb(q)
        if QE is not None:
            re_, ce_ = [], []
            for a in range(0, len(qe_rows), 256):
                top = (QE[a:a + 256] @ self.E.T).topk(K_EMB, dim=1).indices.cpu().numpy()
                re_.append(np.repeat(qe_rows[a:a + 256], K_EMB))
                ce_.append(top.ravel())
            keys.append(np.concatenate(re_) * (1 << 32) + np.concatenate(ce_))
        key = np.unique(np.concatenate(keys))
        qi, si = key >> 32, key & ((1 << 32) - 1)
        t1 = time.time()
        Fqn, Fqa = self.fn.transform(_pad(q["name_ns"])), self.fa.transform(_pad(q["addr"]))
        cols = {"q_row": qi.astype(np.uint32), "s_row": si.astype(np.uint32),
                "cos_name": _rowdot(Fqn, self.Fn, qi, si), "cos_addr": _rowdot(Fqa, self.Fa, qi, si),
                "cos_addr_w": _rowdot(Aa, self.Ba, qi, si), "cos_comb_w": _rowdot(Ac, self.Bc, qi, si),
                "from_name": np.isin(key, keys[0]), "from_addr": np.isin(key, keys[1]),
                "from_comb": np.isin(key, keys[2]),
                "from_emb": np.isin(key, keys[3]) if len(keys) > 3 else np.zeros(len(key), bool)}
        emb_cos = np.full(len(key), np.nan, np.float32)
        if QE is not None:
            loc = np.full(q.height, -1)
            loc[qe_rows] = np.arange(len(qe_rows))
            m = loc[qi] >= 0
            nz = np.flatnonzero(m)
            for a in range(0, len(nz), 200_000):
                sel = nz[a:a + 200_000]
                emb_cos[sel] = (QE[loc[qi[sel]]] * self.E[si[sel]]).float().sum(1).cpu().numpy()
        cols["emb_cos"] = emb_cos
        p = pl.DataFrame(cols).join(q, on="q_row").join(self.s1, on="s_row", suffix="_s")
        p = add_features(p)
        log(f"    search {t1 - t0:.0f}s, features {time.time() - t1:.0f}s, {p.height} pairs")
        return p


def _margin(p, c, grp="q_row"):
    """value minus the best OTHER value in the group (positive only for a clear winner)."""
    g = p.group_by(grp).agg(pl.col(c).max().alias("_m1"), pl.col(c).top_k(2).min().alias("_m2"))
    p = p.join(g, on=grp)
    other = pl.when(pl.col(c) == pl.col("_m1")).then(pl.col("_m2")).otherwise(pl.col("_m1"))
    other = pl.when(pl.len().over(grp) == 1).then(0.0).otherwise(other)
    return p.with_columns((pl.col(c) - other).alias(f"{c}_margin")).drop("_m1", "_m2")


def add_features(p):
    qn, sn = p["name_core"].to_list(), p["name_core_s"].to_list()
    qs, ss = p["name_ns"].to_list(), p["name_ns_s"].to_list()
    qa, sa = p["addr"].to_list(), p["addr_s"].to_list()
    qf = p["addr_nums"].list.first().fill_null("").to_list()
    sf = p["addr_nums_s"].list.first().fill_null("").to_list()
    f = {
        "n_ratio": _fuzz(qn, sn, fuzz.ratio),
        "n_tset": _fuzz(qn, sn, fuzz.token_set_ratio),
        "n_tsort": _fuzz(qn, sn, fuzz.token_sort_ratio),
        "n_partial": _fuzz(qs, ss, fuzz.partial_ratio),
        "n_jw": _fuzz(qs, ss, JaroWinkler.normalized_similarity),
        "n_full_ratio": _fuzz(p["name_full"].to_list(), p["name_full_s"].to_list(), fuzz.ratio),
        "a_ratio": _fuzz(qa, sa, fuzz.ratio),
        "a_tset": _fuzz(qa, sa, fuzz.token_set_ratio),
        "a_partial": _fuzz(qa, sa, fuzz.partial_ratio),
        "num_first_lev": _fuzz(qf, sf, Levenshtein.distance),
    }
    del qn, sn, qs, ss, qa, sa
    p = p.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    q1, s1 = pl.col("addr_nums").list.first(), pl.col("addr_nums_s").list.first()
    q1f, s1f = q1.cast(pl.Float64, strict=False), s1.cast(pl.Float64, strict=False)
    inter_al = pl.col("alnum").list.set_intersection("alnum_s").list.len()
    p = p.with_columns(
        num_q=pl.col("addr_nums").list.len().cast(pl.Float32),
        num_s=pl.col("addr_nums_s").list.len().cast(pl.Float32),
        num_common=pl.col("addr_nums").list.set_intersection("addr_nums_s").list.len().cast(pl.Float32),
        num_first_eq=(q1 == s1).fill_null(False),
        num_first_in=pl.col("addr_nums_s").list.contains(q1).fill_null(False),
        num_first_prefix=((q1 != s1) & (s1.str.starts_with(q1) | q1.str.starts_with(s1))).fill_null(False),
        num_first_reldiff=((q1f - s1f).abs() / pl.max_horizontal(q1f, s1f, pl.lit(1.0))).cast(pl.Float32),
        num_first_lev=pl.when(q1.is_null() | s1.is_null()).then(None).otherwise(pl.col("num_first_lev")),
        alnum_common=inter_al.cast(pl.Float32),
        alnum_q_in_s=(inter_al / pl.col("alnum").list.len()).cast(pl.Float32),
        len_nq=pl.col("name_core").str.len_chars().cast(pl.Float32),
        len_ns=pl.col("name_core_s").str.len_chars().cast(pl.Float32),
        len_aq=pl.col("addr").str.len_chars().cast(pl.Float32),
        len_as=pl.col("addr_s").str.len_chars().cast(pl.Float32),
        name_exact=(pl.col("name_core") == pl.col("name_core_s")),
    )
    # competition among the candidates of the same S2/S3 row
    for c in ["cos_name", "cos_addr", "n_tset", "a_tset", "emb_cos", "cos_comb_w"]:
        p = _margin(p, c)
        p = p.with_columns(pl.col(c).rank("ordinal", descending=True).over("q_row").cast(pl.Float32).alias(f"{c}_rank"))
    p = p.with_columns(emb_cos_rank=pl.when(pl.col("emb_cos").is_nan()).then(None).otherwise(pl.col("emb_cos_rank")))
    p = p.with_columns(n_cand=pl.len().over("q_row").cast(pl.Float32),
                       n_name_hi=(pl.col("n_tset") >= 90).sum().over("q_row").cast(pl.Float32),
                       n_addr_hi=(pl.col("a_tset") >= 90).sum().over("q_row").cast(pl.Float32))
    return p.select(["entity_id", "entity_id_s"] + FEATURES)


FEATURES = ["cos_name", "cos_addr", "cos_addr_w", "cos_comb_w", "from_name", "from_addr", "from_comb",
            "from_emb", "emb_cos",
            "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_full_ratio",
            "a_ratio", "a_tset", "a_partial",
            "num_q", "num_s", "num_common", "num_first_eq", "num_first_in", "num_first_prefix",
            "num_first_reldiff", "num_first_lev", "alnum_common", "alnum_q_in_s",
            "len_nq", "len_ns", "len_aq", "len_as", "name_exact", "s_name_count",
            "src", "name_is_domain", "name_has_alias", "name_nonlatin", "addr_missing", "addr_nonlatin",
            "cos_name_margin", "cos_name_rank", "cos_addr_margin", "cos_addr_rank",
            "n_tset_margin", "n_tset_rank", "a_tset_margin", "a_tset_rank",
            "emb_cos_margin", "emb_cos_rank", "cos_comb_w_margin", "cos_comb_w_rank",
            "n_cand", "n_name_hi", "n_addr_hi"]


def country_pairs(s1, q, split, on_chunk, skip=None):
    """Build the index for one country and call on_chunk(pairs_df, chunk_no) for every chunk of S2/S3 rows.
    skip(i) -> True skips chunks already on disk (resume after an interruption)."""
    idx = CountryIndex(s1, split)
    for i, a in enumerate(range(0, q.height, CHUNK)):
        if skip is not None and skip(i):
            log(f"  chunk {i}: already on disk, skipped")
            continue
        on_chunk(idx.chunk_pairs(q.slice(a, CHUNK)), i)
        log(f"  chunk {i}: {min(a + CHUNK, q.height)}/{q.height} rows")
