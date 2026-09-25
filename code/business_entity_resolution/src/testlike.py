"""Derive a training set that looks like TEST from the full-train features (no rebuild).

Test differs from train (EXPERIMENTS.md, test-side EDA): fewer S1 rows per country (US 663k vs 1.32M)
and more unmatched look-alike S2/S3 records per S1 (S2/S3 per S1: test ~5.8 vs train 4.68).
We (1) keep a random share of S1 rows per country to reach test density, (2) let records whose
business was dropped become unmatched look-alikes, (3) sample records so S2/S3-per-S1 matches test,
(4) drop pairs to removed S1 rows and recompute every feature that depends on the other candidates.

Usage: python testlike.py            -> WORK/pairs/tlike/   then: pipeline.py train tlike sib
"""
import glob
import os
import zlib
import polars as pl
from common import WORK, log, read_truth, id_to_int
from candidates import _margin, FEATURES

SRC, DST = os.path.join(WORK, "pairs", "full"), os.path.join(WORK, "pairs", "tlike")
TEST_S1 = {"US": 663_106, "India": 809_986}
TEST_Q_PER_S1 = {"US": 5.756, "India": 5.824}


def main():
    os.makedirs(DST, exist_ok=True)
    s1 = pl.read_parquet(os.path.join(SRC, "s1.parquet"))
    q = pl.read_parquet(os.path.join(SRC, "q.parquet"))
    n = s1.group_by("country").len()
    keep_rate = {c: min(1.0, TEST_S1[c] / k) for c, k in n.iter_rows()}
    log("S1 keep rate:", keep_rate)
    h = pl.col("s1_id").map_elements(lambda x: zlib.crc32(("tl" + x).encode()) % 100000, return_dtype=pl.Int64)
    s1 = s1.with_columns(h=h).filter(pl.col("h") < pl.col("country").replace_strict(keep_rate) * 100000).drop("h")
    truth = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    matched_kept = truth.join(s1.select("s"), on="s").select("q")
    qk = []
    for c in keep_rate:
        qc = q.filter(pl.col("country") == c)
        m = qc.join(matched_kept, on="q")
        pool = qc.join(matched_kept, on="q", how="anti")          # unmatched + records whose business was dropped
        need = int(TEST_Q_PER_S1[c] * s1.filter(pl.col("country") == c).height) - m.height
        rate = min(1.0, need / pool.height)
        d = pool.filter((pl.col("q").hash(seed=11) % 100000) < rate * 100000)
        log(f"{c}: S1 kept {s1.filter(pl.col('country') == c).height}, matched records {m.height}, "
            f"look-alike records {d.height} (pool {pool.height}), S2/S3 per S1 "
            f"{(m.height + d.height) / s1.filter(pl.col('country') == c).height:.2f}")
        qk += [m, d]
    q = pl.concat(qk)
    s1.write_parquet(os.path.join(DST, "s1.parquet"))
    q.write_parquet(os.path.join(DST, "q.parquet"))

    # chain count at test density: S1 rows sharing the core name among KEPT S1 rows
    names = pl.concat([pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "name_core"])
                       .select(s=id_to_int("entity_id"), nc="name_core")]).join(s1.select("s"), on="s")
    chain = names.with_columns(s_name_count2=pl.len().over("nc").cast(pl.Float32)).select("s", "s_name_count2")
    qs, ss = q.select("q"), s1.select("s")
    for f in sorted(glob.glob(os.path.join(SRC, "*_*.parquet"))):
        p = pl.read_parquet(f).join(qs, on="q").join(ss, on="s")
        p = p.join(chain, on="s", how="left").with_columns(s_name_count=pl.col("s_name_count2")).drop("s_name_count2")
        for col in ["cos_name", "cos_addr", "n_tset", "a_tset", "emb_cos", "cos_comb_w"]:
            p = _margin(p.drop(f"{col}_margin"), col, grp="q")
            p = p.with_columns(pl.col(col).rank("ordinal", descending=True).over("q").cast(pl.Float32).alias(f"{col}_rank"))
        p = p.with_columns(emb_cos_rank=pl.when(pl.col("emb_cos").is_nan()).then(None).otherwise(pl.col("emb_cos_rank")),
                           n_cand=pl.len().over("q").cast(pl.Float32),
                           n_name_hi=(pl.col("n_tset") >= 90).sum().over("q").cast(pl.Float32),
                           n_addr_hi=(pl.col("a_tset") >= 90).sum().over("q").cast(pl.Float32))
        p.select(["q", "s"] + [pl.col(x).cast(pl.Float32) for x in FEATURES]).write_parquet(
            os.path.join(DST, os.path.basename(f)))
        log(f"{os.path.basename(f)}: {p.height} pairs")
    for c in keep_rate:
        open(os.path.join(DST, f"{c}.done"), "w").close()


if __name__ == "__main__":
    main()
