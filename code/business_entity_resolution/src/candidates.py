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

BER_V8=1 switches on the v8 blocking (audit 26 Sep, EXPERIMENTS.md "Blocking and cleaning audit"): name search
max_df 4000 -> 10000, no-address name k 30 -> 100, word searches with min_df 1 (hashed, HashTfidf) and the dense e5
search for EVERY record (embed.py encode_all). Defaults (unset) are the production settings of v2-v7.
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

V8 = os.environ.get("BER_V8") == "1"
K_NAME = 10
K_NAME_NOADDR = 100 if V8 else 30     # v8: +0.16 US / +0.20 India recall for ~3 candidates/record
K_ADDR = 10
K_COMB = 20
K_EMB = 10
MAX_DF_NAME = 10000 if V8 else 4000   # v8: at 4000 a median US name kept 4 3-grams at full density, 3.2% none
MAX_DF_WORD = 5000
CHUNK = int(os.environ.get("BER_CHUNK", "250000"))
THREADS = int(os.environ.get("BER_THREADS", "11"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class HashTfidf:
    """v8 word 1-2-gram TF-IDF with min_df=1 (every S1 term kept; +0.17 US / +0.18 India recall at no candidate
    cost) without a vocabulary dict, which would be huge at min_df=1: hashed terms, sublinear tf x smooth idf fitted
    on S1, terms in more than max_df S1 rows or in none dropped, L2-normalised rows (as sklearn)."""

    def __init__(self, max_df, nf=1 << 24):
        """Hashed word 1-2-gram vectoriser (2**24 buckets); terms in more than max_df S1 rows get weight 0."""
        from sklearn.feature_extraction.text import HashingVectorizer
        self.hv = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False,
                                    n_features=nf, alternate_sign=False, norm=None, dtype=np.float32)
        self.max_df = max_df

    def fit_transform(self, texts):
        """Fit the idf on the S1 texts; return their weighted, L2-normalised rows."""
        S = self.hv.transform(texts).tocsr()
        n = S.shape[0]
        df = np.bincount(S.indices, minlength=S.shape[1])
        self.idf = ((np.log((1 + n) / (1 + df)) + 1) * ((df >= 1) & (df <= self.max_df))).astype(np.float32)
        return self._weigh(S)

    def transform(self, texts):
        """Weighted, L2-normalised rows of record texts, with the idf fitted on S1."""
        return self._weigh(self.hv.transform(texts).tocsr())

    def _weigh(self, M):
        """Sublinear tf x idf, zero weights removed, rows L2-normalised (modifies M)."""
        M.data = ((1 + np.log(M.data)) * self.idf[M.indices]).astype(np.float32)
        M.eliminate_zeros()
        nr = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
        nr[nr == 0] = 1
        M.data /= np.repeat(nr, np.diff(M.indptr)).astype(np.float32)
        return M

S_COLS = ["entity_id", "name_full", "name_core", "name_ns", "addr", "addr_nums", "business_address"]
Q_COLS = S_COLS + ["src", "name_is_domain", "name_has_alias", "name_nonlatin", "addr_missing", "addr_nonlatin"]
_ALNUM = r"[a-z0-9]*\d[a-z0-9]*(?:[/\-][a-z0-9]+)*"


def _vec(max_df):
    """TF-IDF over character 3-grams (min_df 2, sublinear tf); max_df caps a term's document frequency
    (1.0 = no cap)."""
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32, lowercase=False)


def _wvec(max_df):
    """TF-IDF over word 1-2-grams of whitespace tokens (min_df 2, sublinear tf), document-frequency cap max_df."""
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32, lowercase=False)


def _comb(df):
    """Text of the combined search, one string per row: core name + ' ' + cleaned address."""
    return (df["name_core"] + " " + df["addr"]).to_list()


def _pad(series):
    """Pad each string with spaces, so the first and last character 3-grams mark the word boundaries."""
    return (" " + series + " ").to_list()


def _topn(A, B, k):
    """Top-k cosine neighbours in B of each row of A (sparse product; scores below 0.01 dropped).
    Returns the (row, column) index arrays."""
    C = sp_matmul_topn(A, B, top_n=k, threshold=0.01, n_threads=THREADS).tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64)


def _rowdot(A, B, qi, si):
    """Cosine of row qi[i] of A with row si[i] of B for every pair i, 2M pairs at a time."""
    out = np.empty(len(qi), np.float32)
    for a in range(0, len(qi), 2_000_000):
        b = a + 2_000_000
        out[a:b] = np.asarray(A[qi[a:b]].multiply(B[si[a:b]]).sum(1)).ravel()
    return out


def _fuzz(a, b, scorer):
    """rapidfuzz `scorer` applied to the string pairs (a[i], b[i]), as float32."""
    return cpdist(a, b, scorer=scorer, workers=int(os.environ.get("BER_THREADS", "-1")), dtype=np.float32)


def _alnum(col):
    """Polars expression: the unique lowercased tokens of an address that contain a digit, such as '8-9-1/14a'."""
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"\s*([/\-])\s*", "$1")
              .str.extract_all(_ALNUM).list.unique())


class CountryIndex:
    """Holds the S1 rows of one country, fitted vectorisers and (optionally) S1 name embeddings."""

    def __init__(self, s1, split):
        """Fit the search vectorisers and the full-vocabulary feature vectorisers on the S1 rows of one country, and
        load the S1 embeddings of `split` when WORK/emb holds them."""
        self.s1 = (s1.select(S_COLS).with_row_index("s_row")
                     .with_columns(s_name_count=pl.len().over("name_core").cast(pl.Float32),
                                   alnum=_alnum("business_address")))
        self.vn = _vec(MAX_DF_NAME)
        self.va, self.vc = ((HashTfidf(MAX_DF_WORD), HashTfidf(MAX_DF_WORD)) if V8
                            else (_wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)))
        self.Bn = self.vn.fit_transform(_pad(self.s1["name_ns"]))
        self.Ba = self.va.fit_transform(self.s1["addr"].to_list())
        self.Bc = self.vc.fit_transform(_comb(self.s1))
        # combined SEARCH (which S1 rows become candidates); the cos_comb_w feature always uses vc/Bc.
        # A subclass may widen it (blocking_b2.py: max_df 20000, top 40).
        self.vcs, self.Bcs, self.kc = self.vc, self.Bc, K_COMB
        self.fn, self.fa = _vec(1.0), _vec(1.0)
        self.Fn = self.fn.fit_transform(_pad(self.s1["name_ns"]))
        self.Fa = self.fa.fit_transform(_pad(self.s1["addr"]))
        self.E, self.qemb = None, None
        path = os.path.join(WORK, "emb", f"{split}_s1.npz")
        allq = os.path.join(WORK, "emb", f"{split}_qall_v.npy")
        if V8 and os.path.exists(allq):
            # every S2/S3 record has an embedding (embed.py encode_all): vectors memory-mapped and joined per chunk,
            # so a build process does not hold 10M vectors and an id dict in RAM
            z = np.load(os.path.join(WORK, "emb", f"{split}_s1all.npz"))
            pos = pl.DataFrame({"entity_id": z["ids"], "erow": np.arange(len(z["ids"]))})
            er = self.s1.select("entity_id").join(pos, on="entity_id", how="left", maintain_order="left")["erow"]
            assert er.null_count() == 0, "S1 rows without an embedding"
            self.E = torch.tensor(z["v"][er.to_numpy()], device=DEV)
            del z
            self.qemb = (pl.DataFrame({"entity_id": np.load(os.path.join(WORK, "emb", f"{split}_qall_ids.npy"),
                                                           allow_pickle=True)}).with_row_index("vrow"),
                         np.load(allq, mmap_mode="r"))
        elif os.path.exists(path):
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
        if isinstance(pos, pl.DataFrame):
            m = q.select("entity_id").with_row_index("qi").join(pos, on="entity_id").sort("qi")
            if m.height == 0:
                return None, None
            qi, vi = m["qi"].to_numpy().astype(np.int64), m["vrow"].to_numpy().astype(np.int64)
            return qi, torch.tensor(np.asarray(V[vi]), device=DEV)
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
        r3, c3 = _topn(Ac if self.vcs is self.vc else self.vcs.transform(_comb(q)), self.Bcs, self.kc)
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
    """The 55 pair features (FEATURES) of candidate pairs p, which hold the record's columns and the S1 row's
    columns (suffix _s): name and address similarities, house-number agreement, lengths, flags, and the margin and rank of
    six similarities among the record's candidates. Returns entity_id, entity_id_s and FEATURES."""
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
