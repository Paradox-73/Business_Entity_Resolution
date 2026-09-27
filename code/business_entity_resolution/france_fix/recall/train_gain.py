import os, sys, zlib
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, read_truth, id_to_int, log, macro_f05_df
from pipeline import decide_expf
exec(open(r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/gap/train_analog.py").read().split("s1 = pl.read_parquet")[0].split("G = os.path")[1].join(["G = os.path", ""]) if False else "")
G = os.path.join(WORK, "gathik", "v9")
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
def held(oof_path, s3_path):
    oof = i64(pl.read_parquet(oof_path, columns=["q", "s", "p2"])); s3 = i64(pl.read_parquet(s3_path, columns=["q", "s", "p3"]))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", p=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0)).otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))
s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]).select(s=id_to_int("entity_id").cast(pl.Int64),
     h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
evs = s1.filter(pl.col("h") < 500).select("s")
truth = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)).join(evs, on="s")
sm = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_bgefolds.parquet"))
po = decide_expf(sm.rename({"p": "p2"}), "p2", 0.5, 1.0).join(evs, on="s")
a = pl.read_parquet(r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/gap/train_analog.parquet")
base = macro_f05_df(po, truth, evs)
N = evs.height
for name, sub in [("all", a), ("pg>=0.9", a.filter(pl.col("pg") >= 0.9)),
                  ("no word-level change", a.filter(~pl.col("nm").str.contains("n_swap|n_add|n_drop")))]:
    f = macro_f05_df(pl.concat([po, sub.select("s", "q")]), truth, evs)
    log(f"{name:24s} n {sub.height:6d} prec {sub['y'].mean():.4f}  dF {f - base:+.5f}  per pair x N {(f - base) * N / sub.height:+.4f}")
