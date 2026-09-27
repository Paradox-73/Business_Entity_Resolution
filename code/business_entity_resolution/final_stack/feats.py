"""Shared per-pair feature construction for held-out and test (identical definitions)."""
import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int

OUT = r"C:/ber_scratch/final2/usi-stack"
FEATS = ["a", "b", "c", "po", "pg", "p", "op1", "op2", "gp1", "gp2", "has_o", "has_g", "t_o", "t_g",
         "rank_q", "margin_q", "nq", "rank_s", "margin_s", "ctry", "q_noaddr", "s_noaddr", "hn_d", "hn_abs", "name_eq",
         "d_og", "d_ab"]


def i64(df):
    return df.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def rec_info(split):
    parts = []
    for k in (2, 3):
        parts.append(pl.scan_parquet(os.path.join(WORK, f"{split}_s{k}.parquet")).select(
            q=id_to_int("entity_id").cast(pl.Int64), q_noaddr=pl.col("addr_missing").cast(pl.Int8),
            q_hn=pl.col("addr_nums").list.first().cast(pl.Float64, strict=False),
            q_nh=pl.col("name_core").hash()))
    return pl.concat(parts).collect()


def s1_info(split, countries=("US", "India")):
    return (pl.scan_parquet(os.path.join(WORK, f"{split}_s1.parquet")).filter(pl.col("country").is_in(list(countries)))
            .select(s=id_to_int("entity_id").cast(pl.Int64), s1_id="entity_id", ctry=(pl.col("country") == "India").cast(pl.Int8),
                    s_noaddr=pl.col("addr_missing").cast(pl.Int8),
                    s_hn=pl.col("addr_nums").list.first().cast(pl.Float64, strict=False), s_nh=pl.col("name_core").hash())
            .collect())


def blend_p(d):
    """d has a (small), b (bge), pg (gathik). po = 0.3a+0.7b normalised; p = 0.6 po + 0.4 pg where both."""
    d = d.with_columns(po=pl.when(pl.col("a").is_null() & pl.col("b").is_null()).then(None).otherwise((0.3 * pl.col("a").fill_null(0) + 0.7 * pl.col("b").fill_null(0)) /
                           (pl.when(pl.col("a").is_null()).then(0.0).otherwise(0.3) +
                            pl.when(pl.col("b").is_null()).then(0.0).otherwise(0.7))).cast(pl.Float32))
    return d.with_columns(p=pl.when(pl.col("pg").is_null()).then(pl.col("po")).when(pl.col("po").is_null()).then(pl.col("pg"))
                          .otherwise(0.4 * pl.col("pg") + 0.6 * pl.col("po")).cast(pl.Float32))


def add_feats(d, s1, rec):
    """d: q, s, a, b, c, op1, op2, pg, gp1, gp2 (+ p/po from blend_p)."""
    d = d.with_columns(has_o=pl.col("po").is_not_null().cast(pl.Int8), has_g=pl.col("pg").is_not_null().cast(pl.Int8),
                       t_o=((pl.col("b") - pl.col("op2")).abs() > 1e-5).cast(pl.Int8),
                       t_g=((pl.col("pg") - pl.col("gp2")).abs() > 1e-5).cast(pl.Int8),
                       d_og=(pl.col("po") - pl.col("pg")).cast(pl.Float32), d_ab=(pl.col("a") - pl.col("b")).cast(pl.Float32))
    d = d.with_columns(rank_q=pl.col("p").rank("ordinal", descending=True).over("q").cast(pl.Int16),
                       nq=pl.len().over("q").cast(pl.Int16),
                       rank_s=pl.col("p").rank("ordinal", descending=True).over("s").cast(pl.Int16))
    # margin vs best other candidate in the same record / same S1
    d = d.with_columns(m1q=pl.col("p").max().over("q"), m2q=pl.col("p").top_k(2).min().over("q"),
                       m1s=pl.col("p").max().over("s"), m2s=pl.col("p").top_k(2).min().over("s"))
    d = d.with_columns(margin_q=pl.when(pl.col("rank_q") == 1).then(pl.col("p") - pl.when(pl.col("nq") > 1).then(pl.col("m2q")).otherwise(0.0))
                       .otherwise(pl.col("p") - pl.col("m1q")).cast(pl.Float32),
                       margin_s=pl.when(pl.col("rank_s") == 1).then(pl.col("p") - pl.col("m2s")).otherwise(pl.col("p") - pl.col("m1s")).cast(pl.Float32)
                       ).drop("m1q", "m2q", "m1s", "m2s")
    d = d.join(s1.select("s", "ctry", "s_noaddr", "s_hn", "s_nh"), on="s", how="inner")
    d = d.join(rec, on="q", how="left")
    d = d.with_columns(hn_d=(pl.col("q_hn") - pl.col("s_hn")).cast(pl.Float32),
                       name_eq=(pl.col("q_nh") == pl.col("s_nh")).cast(pl.Int8))
    d = d.with_columns(hn_abs=pl.col("hn_d").abs()).drop("q_hn", "s_hn", "q_nh", "s_nh")
    return d
