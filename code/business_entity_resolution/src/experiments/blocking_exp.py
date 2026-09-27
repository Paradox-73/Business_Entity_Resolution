"""Blocking experiment at FULL density: recall and time of candidate searches for one country.

Usage: python blocking_exp.py US 50000
Queries = sample of S2/S3 rows whose true S1 is in the eval half; index = ALL train S1 rows of that country.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import WORK, log, read_truth, id_to_int
import os

try:
    import torch
except ImportError:
    torch = None


DEVICE = torch.device(
    "cuda") if torch is not None and torch.cuda.is_available() else None
GPU_BATCH = 32


def tfidf(analyzer, ngram, max_df, token_pattern=None):
    kw = dict(analyzer=analyzer, ngram_range=ngram, min_df=2, max_df=max_df, sublinear_tf=True,
              dtype=np.float32, lowercase=False)
    if token_pattern:
        kw["token_pattern"] = token_pattern
    return TfidfVectorizer(**kw)


def _torch_csr(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
        device=DEVICE,
    )


def _gpu_search(A, B, k):
    B_gpu = _torch_csr(B.T)
    rows, cols, values = [], [], []
    top_k = min(k, B.shape[0])
    with torch.no_grad():
        for start in range(0, A.shape[0], GPU_BATCH):
            end = start + GPU_BATCH
            scores = torch.sparse.mm(_torch_csr(
                A[start:end]), B_gpu).to_dense()
            values_batch, cols_batch = scores.topk(top_k, dim=1)
            keep = values_batch >= 0.01
            row_batch = torch.arange(scores.shape[0], device=DEVICE).unsqueeze(
                1).expand_as(cols_batch)
            rows.append((row_batch[keep] + start).cpu().numpy())
            cols.append(cols_batch[keep].cpu().numpy())
            values.append(values_batch[keep].cpu().numpy())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(values)


def search(vec, s_text, q_text, k):
    B = vec.fit_transform(s_text)
    t0 = time.time()
    A = vec.transform(q_text)
    if DEVICE is not None:
        r, c, _ = _gpu_search(A, B, k)
        return r, c, time.time() - t0
    C = sp_matmul_topn(A, B, top_n=k, threshold=0.01, n_threads=11).tocoo()
    return C.row, C.col, time.time() - t0


def main(country="US", n=50000):
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=[
                         "entity_id", "name_ns", "name_core", "addr", "addr_nums", "addr_missing", "country"])
    s1 = s1.filter(pl.col("country") == country).with_row_index("s_row")
    truth = read_truth()
    q = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"),
                                   columns=["entity_id", "name_ns", "name_core", "addr", "addr_nums", "addr_missing", "country"]) for k in (2, 3)])
    q = q.filter(pl.col("country") == country).join(
        truth.rename({"q_id": "entity_id"}), on="entity_id")
    q = q.sample(n, seed=0).join(
        s1.select(s1_id="entity_id", true_row="s_row"), on="s1_id")
    log(f"search device: {DEVICE or 'cpu'}")
    log(f"{country}: S1 {s1.height}, queries {q.height}")
    tr = q["true_row"].to_numpy()
    full_tr = tr
    found = {}

    def rec(name, r, c, secs):
        hit = np.zeros(len(tr), bool)
        hit[r[c == tr[r]]] = True
        found[name] = hit
        log(f"  {name:40s} recall {hit.mean():.4f}  pairs/q {len(r) / len(tr):5.1f}  search {secs:5.1f}s")

    def pad(s): return (" " + s + " ").to_list()
    W = r"\S+"
    def comb(d): return (d["name_core"] + " " + d["addr"]).to_list()
    r, c, t = search(tfidf("char", (3, 3), 4000), pad(
        s1["name_ns"]), pad(q["name_ns"]), 10)
    rec("name char3 4000 k10", r, c, t)
    for k in (10, 20):
        r, c, t = search(tfidf("word", (1, 2), 5000, W),
                         s1["addr"].to_list(), q["addr"].to_list(), k)
        rec(f"addr word 5000 k{k}", r, c, t)
        r, c, t = search(tfidf("word", (1, 2), 5000, W),
                         s1["name_core"].to_list(), q["name_core"].to_list(), k)
        rec(f"name word 5000 k{k}", r, c, t)
        r, c, t = search(tfidf("word", (1, 2), 5000, W), comb(s1), comb(q), k)
        rec(f"name+addr word 5000 k{k}", r, c, t)
    noaddr = q.filter(pl.col("addr_missing"))
    noaddr_tr = noaddr["true_row"].to_numpy()
    tr = noaddr_tr
    for k in (50, 100):
        r, c, t = search(tfidf("word", (1, 2), 5000, W),
                         s1["name_core"].to_list(), noaddr["name_core"].to_list(), k)
        rec(f"name word 5000 noaddr k{k}", r, c, t)
    tr = full_tr
    r, c, t = search(tfidf("word", (1, 2), 20000, W), comb(s1), comb(q), 20)
    rec("name+addr word 20000 k20", r, c, t)
    for combo in (["name char3 4000 k10", "addr word 5000 k10", "name word 5000 k10"],
                  ["name char3 4000 k10", "addr word 5000 k10",
                      "name word 5000 k10", "name+addr word 5000 k10"],
                  ["name char3 4000 k10", "addr word 5000 k20",
                      "name word 5000 k20", "name+addr word 5000 k20"],
                  ["name char3 4000 k10", "addr word 5000 k10",
                      "name+addr word 5000 k20"],
                  ["name char3 4000 k10", "name+addr word 5000 k20"],
                  ["name char3 4000 k10", "addr word 5000 k10", "name word 5000 k10", "name+addr word 20000 k20"]):
        u = np.any(np.stack([found[x] for x in combo]), axis=0)
        log(f"  COMBO {combo}: {u.mean():.4f}")
    found.clear()
    keys = list(found)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            u = found[a] | found[b]
            if u.mean() > 0.9:
                log(f"  UNION {a} + {b}: {u.mean():.4f}")
    allu = np.any(np.stack(list(found.values())), axis=0)
    log(f"  UNION of all: {allu.mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
