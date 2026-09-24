"""Blocking (shortlist) + pair features, one country at a time, Source 2/3 rows in chunks.

Blocking, searched from each S2/S3 row towards S1 rows of the SAME country:
  - name:    TF-IDF over character 3-grams of the space-free core name, top K_NAME by cosine
  - address: TF-IDF over character 3-grams of the cleaned address,      top K_ADDR by cosine
The union of both lists is the candidate set scored by the model.
"""
import time
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import log

K_NAME = 10
K_ADDR = 10
MAX_DF_NAME = 4000
MAX_DF_ADDR = 2000
CHUNK = 250_000
THREADS = 11

S_COLS = ["entity_id", "name_full", "name_core", "name_ns", "addr", "addr_nums"]
Q_COLS = S_COLS + ["src", "name_is_domain", "name_has_alias", "name_nonlatin", "addr_missing", "addr_nonlatin"]


def _vec(max_df):
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32, lowercase=False)


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


class CountryIndex:
    """Holds the S1 rows of one country and the fitted vectorisers."""

    def __init__(self, s1):
        self.s1 = s1.select(S_COLS).with_row_index("s_row")
        # chain-name risk: how many S1 rows in this country share the exact core name
        self.s1 = self.s1.with_columns(s_name_count=pl.len().over("name_core").cast(pl.Float32))
        # absolute document-frequency caps: a 3-gram shared by more S1 rows than this is dropped.
        # Keeps search cost per query bounded as S1 grows (US test S1 = 663k rows).
        self.vn, self.va = _vec(MAX_DF_NAME), _vec(MAX_DF_ADDR)
        self.Bn = self.vn.fit_transform(_pad(self.s1["name_ns"]))
        self.Ba = self.va.fit_transform(_pad(self.s1["addr"]))
        log(f"  S1 index: {self.s1.height} rows, name vocab {self.Bn.shape[1]}, addr vocab {self.Ba.shape[1]}")

    def chunk_pairs(self, q):
        """Candidate pairs + features for a chunk of S2/S3 rows (DataFrame with Q_COLS)."""
        q = q.select(Q_COLS).with_row_index("q_row")
        An = self.vn.transform(_pad(q["name_ns"]))
        Aa = self.va.transform(_pad(q["addr"]))
        t0 = time.time()
        r1, c1 = _topn(An, self.Bn, K_NAME)
        r2, c2 = _topn(Aa, self.Ba, K_ADDR)
        log(f"    search {time.time() - t0:.0f}s")
        key = np.unique(np.concatenate([r1, r2]) * (1 << 32) + np.concatenate([c1, c2]))
        qi, si = key >> 32, key & ((1 << 32) - 1)
        in_name = np.isin(key, r1 * (1 << 32) + c1)
        in_addr = np.isin(key, r2 * (1 << 32) + c2)
        p = pl.DataFrame({"q_row": qi.astype(np.uint32), "s_row": si.astype(np.uint32),
                          "cos_name": _rowdot(An, self.Bn, qi, si), "cos_addr": _rowdot(Aa, self.Ba, qi, si),
                          "from_name": in_name, "from_addr": in_addr})
        p = p.join(q, on="q_row").join(self.s1, on="s_row", suffix="_s")
        return add_features(p)


def add_features(p):
    qn, sn = p["name_core"].to_list(), p["name_core_s"].to_list()
    qs, ss = p["name_ns"].to_list(), p["name_ns_s"].to_list()
    qa, sa = p["addr"].to_list(), p["addr_s"].to_list()
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
    }
    del qn, sn, qs, ss, qa, sa
    p = p.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    qf, sf = pl.col("addr_nums").list.first(), pl.col("addr_nums_s").list.first()
    p = p.with_columns(
        num_q=pl.col("addr_nums").list.len().cast(pl.Float32),
        num_s=pl.col("addr_nums_s").list.len().cast(pl.Float32),
        num_common=pl.col("addr_nums").list.set_intersection("addr_nums_s").list.len().cast(pl.Float32),
        num_first_eq=(qf == sf).fill_null(False),
        num_first_in=pl.col("addr_nums_s").list.contains(qf).fill_null(False),
        num_first_prefix=((qf != sf) & (sf.str.starts_with(qf) | qf.str.starts_with(sf))).fill_null(False),
        len_nq=pl.col("name_core").str.len_chars().cast(pl.Float32),
        len_ns=pl.col("name_core_s").str.len_chars().cast(pl.Float32),
        len_aq=pl.col("addr").str.len_chars().cast(pl.Float32),
        len_as=pl.col("addr_s").str.len_chars().cast(pl.Float32),
        name_exact=(pl.col("name_core") == pl.col("name_core_s")),
    )
    # competition among the candidates of the same S2/S3 row
    for c in ["cos_name", "cos_addr", "n_tset", "a_tset"]:
        p = p.with_columns((pl.col(c) - pl.col(c).max().over("q_row")).alias(f"{c}_gap"),
                           pl.col(c).rank("ordinal", descending=True).over("q_row").cast(pl.Float32).alias(f"{c}_rank"))
    p = p.with_columns(n_cand=pl.len().over("q_row").cast(pl.Float32))
    return p.select(["q_row", "s_row", "entity_id", "entity_id_s"] + FEATURES)


FEATURES = ["cos_name", "cos_addr", "from_name", "from_addr",
            "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_full_ratio",
            "a_ratio", "a_tset", "a_partial",
            "num_q", "num_s", "num_common", "num_first_eq", "num_first_in", "num_first_prefix",
            "len_nq", "len_ns", "len_aq", "len_as", "name_exact", "s_name_count",
            "src", "name_is_domain", "name_has_alias", "name_nonlatin", "addr_missing", "addr_nonlatin",
            "cos_name_gap", "cos_name_rank", "cos_addr_gap", "cos_addr_rank",
            "n_tset_gap", "n_tset_rank", "a_tset_gap", "a_tset_rank", "n_cand"]


def country_pairs(s1, q, on_chunk):
    """Build the index for one country and call on_chunk(pairs_df) for every chunk of S2/S3 rows."""
    idx = CountryIndex(s1)
    for a in range(0, q.height, CHUNK):
        pairs = idx.chunk_pairs(q.slice(a, CHUNK))
        on_chunk(pairs)
        log(f"  chunk {a // CHUNK}: {min(a + CHUNK, q.height)}/{q.height} rows, {pairs.height} pairs")
