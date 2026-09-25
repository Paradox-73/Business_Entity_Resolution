"""Test-like TRAIN split built with the real search (replaces testlike.py, whose v4 lost 0.0086 on the LB).

testlike.py deleted the candidate pairs of dropped S1 rows, which halved US candidate lists (15.9 per record vs
31.3 on test). Here the dropped S1 rows are removed BEFORE the search, so each record's top-k is refilled by
other S1 rows, as at test density. Output: WORK/tl2_s{1,2,3}.parquet and embedding files for split 'tl2';
then build pairs with:  python pipeline.py build tl2 tl2

- S1 rows kept per country at test's count (same hash as testlike.py);
- records of dropped S1 rows stay, now as unmatched look-alikes (test has ~2.3 unmatched records per S1, train 1.2);
- unmatched records sampled so S2/S3 per S1 matches test (US 5.76; India's pool allows ~5.1 of 5.82).
Folds and the eval half are the same as FULL (crc32 of s1_id), so models trained on FULL can be scored on it
out-of-fold (evaluate.py).
"""
import os
import shutil
import zlib
import polars as pl
from common import WORK, log, read_truth

TEST_S1 = {"US": 663_106, "India": 809_986}
TEST_Q_PER_S1 = {"US": 5.756, "India": 5.824}


def main():
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"))
    n = s1.group_by("country").len()
    keep_rate = {c: min(1.0, TEST_S1[c] / k) for c, k in n.iter_rows()}
    h = pl.col("entity_id").map_elements(lambda x: zlib.crc32(("tl" + x).encode()) % 100000, return_dtype=pl.Int64)
    s1 = s1.with_columns(h=h).filter(pl.col("h") < pl.col("country").replace_strict(keep_rate) * 100000).drop("h")
    truth = read_truth()
    matched_kept = truth.join(s1.select(s1_id="entity_id"), on="s1_id").select(entity_id="q_id").unique()
    parts = []
    for k in (2, 3):
        parts.append(pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet")))
    q = pl.concat(parts)
    keep = []
    for c in keep_rate:
        qc = q.filter(pl.col("country") == c)
        m = qc.join(matched_kept, on="entity_id")
        pool = qc.join(matched_kept, on="entity_id", how="anti")
        k1 = s1.filter(pl.col("country") == c).height
        rate = min(1.0, (TEST_Q_PER_S1[c] * k1 - m.height) / pool.height)
        d = pool.filter((pl.col("entity_id").hash(seed=11) % 100000) < rate * 100000)
        log(f"{c}: S1 {k1}, matched records {m.height}, look-alike records {d.height} (pool {pool.height}), "
            f"S2/S3 per S1 {(m.height + d.height) / k1:.2f}")
        keep += [m.select("entity_id"), d.select("entity_id")]
    keep = pl.concat(keep)
    s1.write_parquet(os.path.join(WORK, "tl2_s1.parquet"))
    for k in (2, 3):
        x = parts[k - 2].join(keep, on="entity_id")
        x.write_parquet(os.path.join(WORK, f"tl2_s{k}.parquet"))
        log(f"tl2 S{k}: {x.height} rows")
    for a, b in (("train_s1.npz", "tl2_s1.npz"), ("train_q.npz", "tl2_q.npz")):   # India non-Latin embeddings
        src, dst = os.path.join(WORK, "emb", a), os.path.join(WORK, "emb", b)
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy(src, dst)


if __name__ == "__main__":
    main()
