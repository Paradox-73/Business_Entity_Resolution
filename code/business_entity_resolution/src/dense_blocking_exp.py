"""Dense (bi-encoder) blocking check against the production TF-IDF shortlist, at full train density.

Question: would adding a dense e5 search (for Latin-script names too, not only non-Latin ones) recover true pairs
that the TF-IDF union (candidates.py) misses, and could it run at full scale with FAISS?

Queries: n matched Latin-name S2/S3 records of one country whose true S1 is in the eval half (crc32 % 1000 < 500),
so the fine-tuned e5 (trained on the other half) never saw them. Index: ALL train S1 rows of that country.
Reports:
  1. recall of the production TF-IDF union (same vectorisers, df caps and k as candidates.py)
  2. recall of dense top-k (exact search on the GPU), per model
  3. recall of TF-IDF union + dense top-k, extra candidates per record, and which TF-IDF misses dense recovers
     (no address / chain name = S1 core name shared by several S1 rows)
  4. FAISS IVF (approximate, CPU): recall vs exact and search speed, for the full-scale version

  python dense_blocking_exp.py US 100000
  python dense_blocking_exp.py India 100000
"""
import os
import sys
import time
import zlib
import numpy as np
import polars as pl
import torch
from common import WORK, log, read_truth
from candidates import (_vec, _wvec, _pad, _comb, _topn, K_NAME, K_NAME_NOADDR, K_ADDR, K_COMB,
                        MAX_DF_NAME, MAX_DF_WORD)
from embed import Encoder, text_col

MODELS = [os.path.join(WORK, "e5_ft_addr"), "intfloat/multilingual-e5-small"]
if os.environ.get("BER_DENSE_MODELS") == "ft":      # only the fine-tuned model (halves the run time)
    MODELS = MODELS[:1]
FAISS = os.environ.get("BER_FAISS", "1") == "1"      # exact GPU search is ~1 ms/query; FAISS part is optional
KS = (10, 20, 50)
COLS = ["entity_id", "business_name", "addr", "name_ns", "name_core", "addr_missing", "country"]


def keys(r, c):
    return np.asarray(r, np.int64) * (1 << 32) + np.asarray(c, np.int64)


def load(country, n):
    s1 = (pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=COLS)
            .filter(pl.col("country") == country).with_row_index("s_row")
            .with_columns(chain=pl.len().over("name_core")))
    truth = read_truth()
    ev = truth.select("s1_id").unique().filter(
        pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 < 500, return_dtype=pl.Boolean))
    q = pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet"))
                   .filter((pl.col("country") == country) & ~pl.col("name_nonlatin")).select(COLS).collect()
                   for k in (2, 3)])
    q = (q.join(truth.rename({"q_id": "entity_id"}), on="entity_id").join(ev, on="s1_id")
          .sample(n, seed=0)
          .join(s1.select(s1_id="entity_id", true_row="s_row", true_chain="chain"), on="s1_id"))
    return s1, q


def tfidf_union(s1, q):
    """The production shortlist (candidates.CountryIndex.chunk_pairs minus the non-Latin embedding search)."""
    t0 = time.time()
    vn, va, vc = _vec(MAX_DF_NAME), _wvec(MAX_DF_WORD), _wvec(MAX_DF_WORD)
    Bn, Ba, Bc = vn.fit_transform(_pad(s1["name_ns"])), va.fit_transform(s1["addr"].to_list()), vc.fit_transform(_comb(s1))
    t1 = time.time()
    An, Aa, Ac = vn.transform(_pad(q["name_ns"])), va.transform(q["addr"].to_list()), vc.transform(_comb(q))
    parts = {"name": _topn(An, Bn, K_NAME), "addr": _topn(Aa, Ba, K_ADDR), "comb": _topn(Ac, Bc, K_COMB)}
    noaddr = np.flatnonzero(q["addr_missing"].to_numpy())
    if len(noaddr):
        rr, cc = _topn(An[noaddr], Bn, K_NAME_NOADDR)
        parts["name_noaddr"] = (noaddr[rr], cc)
    log(f"TF-IDF: fit on S1 {t1 - t0:.0f}s, search {time.time() - t1:.0f}s for {q.height} queries")
    return {k: np.unique(keys(*v)) for k, v in parts.items()}


def dense_topk(enc, s1, q, kmax):
    """Exact inner-product top-kmax on the GPU. Returns (S1 vectors fp16 on CPU, top-k indices, encode secs)."""
    t0 = time.time()
    E = enc.encode(s1.select(text_col())["text"].to_list(), bs=512)
    t_s1 = time.time() - t0
    Q = enc.encode(q.select(text_col())["text"].to_list(), bs=512)
    log(f"  encoded S1 {len(E)} in {t_s1:.0f}s ({len(E) / t_s1:.0f}/s), queries {len(Q)}")
    Eg = torch.tensor(E, device="cuda")
    top = np.empty((len(Q), kmax), np.int64)
    t0 = time.time()
    for a in range(0, len(Q), 256):
        qb = torch.tensor(Q[a:a + 256], device="cuda")
        top[a:a + 256] = (qb @ Eg.T).topk(kmax, dim=1).indices.cpu().numpy()
    torch.cuda.synchronize()
    log(f"  exact GPU search {time.time() - t0:.0f}s")
    del Eg
    torch.cuda.empty_cache()
    return E, Q, top


def faiss_check(E, Q, exact_top, true_row, k=20):
    import faiss
    d, n = E.shape[1], len(E)
    nlist = int(4 * np.sqrt(n))
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    rng = np.random.default_rng(0)
    index.train(E[rng.choice(n, min(n, 50 * nlist), replace=False)].astype(np.float32))
    for a in range(0, n, 200_000):
        index.add(E[a:a + 200_000].astype(np.float32))
    log(f"  FAISS IVFFlat nlist={nlist}: train+add {time.time() - t0:.0f}s")
    Qf = Q.astype(np.float32)
    ex = exact_top[:, :k]
    exact_rec = (ex == true_row[:, None]).any(1).mean()

    def report(tag, I, secs):
        overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(I, ex)])
        rec = (I == true_row[:, None]).any(1).mean()
        log(f"  FAISS {tag}: recall@{k} {rec:.4f} (exact {exact_rec:.4f}), overlap with exact top-{k} {overlap:.3f}, "
            f"{secs:.1f}s = {1e6 * secs / len(Q):.0f} us/query")

    for nprobe in (8, 32, 128, 512):
        index.nprobe = nprobe
        t0 = time.time()
        _, I = index.search(Qf, k)
        report(f"IVF nprobe={nprobe:3d}", I, time.time() - t0)
    del index
    # HNSW graph index: usually far better recall than IVF at the same speed, costs a slower build
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    t0 = time.time()
    for a in range(0, n, 200_000):
        hnsw.add(E[a:a + 200_000].astype(np.float32))
    log(f"  FAISS HNSW M=32: build {time.time() - t0:.0f}s")
    for ef in (64, 256):
        hnsw.hnsw.efSearch = ef
        t0 = time.time()
        _, I = hnsw.search(Qf, k)
        report(f"HNSW efSearch={ef:3d}", I, time.time() - t0)
    del hnsw


def main(country="US", n=100_000):
    s1, q = load(country, n)
    if os.environ.get("BER_SMOKE"):     # tiny run to test the code: first 60k S1 rows and their queries
        s1 = s1.head(60_000)
        q = q.filter(pl.col("true_row") < 60_000)
    tr = q["true_row"].to_numpy()
    truek = keys(np.arange(q.height), tr)
    noaddr, chain = q["addr_missing"].to_numpy(), q["true_chain"].to_numpy() > 1
    log(f"{country}: S1 {s1.height}, queries {q.height} (Latin names, true S1 in eval half); "
        f"no address {noaddr.mean():.3f}, true S1 name shared {chain.mean():.3f}")

    tf = tfidf_union(s1, q)
    for k, v in tf.items():
        log(f"  {k:12s} recall {np.isin(truek, v).mean():.4f}")
    U = np.unique(np.concatenate(list(tf.values())))
    hit_tf = np.isin(truek, U)
    log(f"TF-IDF union (production): recall {hit_tf.mean():.4f}, {len(U) / q.height:.1f} candidates/record")
    miss = ~hit_tf
    log(f"  TF-IDF misses {miss.sum()}: no address {noaddr[miss].mean():.3f}, chain name {chain[miss].mean():.3f}, "
        f"both {(noaddr & chain)[miss].mean():.3f}")

    for path in MODELS:
        log(f"MODEL {path}")
        enc = Encoder(path)
        E, Q, top = dense_topk(enc, s1, q, max(KS))
        del enc
        torch.cuda.empty_cache()
        for k in KS:
            dk = keys(np.repeat(np.arange(q.height), k), top[:, :k].ravel())
            hit_d = (top[:, :k] == tr[:, None]).any(1)
            new = np.setdiff1d(np.unique(dk), U)
            both = hit_tf | hit_d
            rec = miss & hit_d
            log(f"  dense@{k:2d}: recall {hit_d.mean():.4f} | union {both.mean():.4f} "
                f"(+{both.mean() - hit_tf.mean():.4f}), +{len(new) / q.height:.1f} new candidates/record | "
                f"recovers {rec.sum()} of {miss.sum()} misses (no address {noaddr[rec].mean() if rec.any() else 0:.2f}, "
                f"chain {chain[rec].mean() if rec.any() else 0:.2f}); dense misses that TF-IDF finds {(hit_tf & ~hit_d).sum()}")
        if path == MODELS[0]:   # TF-IDF misses that dense top-10 finds, for reading
            rec10 = np.flatnonzero(miss & (top[:, :10] == tr[:, None]).any(1))
            ex = q[rec10].select("entity_id", "business_name", "addr", "addr_missing", "true_chain").with_columns(
                s1_name=s1["business_name"].gather(tr[rec10]), s1_addr=s1["addr"].gather(tr[rec10]))
            ex.write_csv(os.path.join(WORK, f"dense_recovered_{country}.tsv"), separator="\t")
            log(f"  wrote dense_recovered_{country}.tsv ({ex.height} rows)")
        if FAISS:
            faiss_check(E, Q, top, tr)
        del E, Q, top


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "US", int(sys.argv[2]) if len(sys.argv) > 2 else 100_000)
