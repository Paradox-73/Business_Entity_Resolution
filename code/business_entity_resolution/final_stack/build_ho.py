"""Held-out feature table: all train US/India pairs, with label, eval flag, cross-fit half. Reproduces 0.99205 baseline."""
import os, sys, zlib
sys.path.insert(0, r"C:/ber_scratch/final2/usi-stack")
from feats import *
from common import read_truth, macro_f05_df, log
from pipeline import decide_expf

G = os.path.join(WORK, "gathik", "v9")


def held(oof_path, s3_path, name):
    oof = i64(pl.read_parquet(oof_path, columns=["q", "s", "p2"]))
    s3 = i64(pl.read_parquet(s3_path, columns=["q", "s", "p3"]))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0))
                    .otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32).alias(name))


sm = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_smallfolds.parquet"), "a")
bg = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_bgefolds.parquet"), "b")
e5 = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_e5lfolds.parquet"), "c")
ours = sm.join(bg, on=["q", "s"], how="full", coalesce=True).join(e5, on=["q", "s"], how="left")
del sm, bg, e5
og = i64(pl.read_parquet(os.path.join(WORK, "models", "full_cons", "oof.parquet"), columns=["q", "s", "p1", "p2"])).rename({"p1": "op1", "p2": "op2"})
ours = ours.join(og.with_columns(pl.col("op1", "op2").cast(pl.Float32)), on=["q", "s"], how="left")
del og
gat = held(os.path.join(G, "models_full_xgb_cons_oof.parquet"), os.path.join(G, "ce_x_oof_s3_bgef0bgef1bgef2.parquet"), "pg")
gg = i64(pl.read_parquet(os.path.join(G, "models_full_xgb_cons_oof.parquet"), columns=["q", "s", "p1", "p2"])).rename({"p1": "gp1", "p2": "gp2"})
gat = gat.join(gg.with_columns(pl.col("gp1", "gp2").cast(pl.Float32)), on=["q", "s"], how="left")
del gg
d = ours.join(gat, on=["q", "s"], how="full", coalesce=True)
del ours, gat
d = blend_p(d)
log(f"pairs {d.height}")
s1 = s1_info("train")
s1 = s1.with_columns(h=pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
rec = rec_info("train")
d = add_feats(d, s1, rec)
del rec
truth = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)).with_columns(y=pl.lit(1, pl.Int8))
d = d.join(truth, on=["s", "q"], how="left").with_columns(pl.col("y").fill_null(0))
d = d.join(s1.select("s", "h"), on="s", how="left")
d = d.with_columns(half=pl.when(pl.col("h") >= 500).then(-1).otherwise(
    pl.col("s").hash(seed=7) % 2).cast(pl.Int8))
d.write_parquet(os.path.join(OUT, "ho.parquet"))
ev = s1.filter(pl.col("h") < 500).select("s", "ctry")
ev = ev.with_columns(half=(pl.col("s").hash(seed=7) % 2).cast(pl.Int8))
ev.write_parquet(os.path.join(OUT, "ev.parquet"))
tr = truth.join(ev.select("s"), on="s").select("s", "q")
tr.write_parquet(os.path.join(OUT, "truth_ev.parquet"))
pred = decide_expf(d.select("q", "s", p2="p"), "p2", 0.5, 1.0).join(ev.select("s"), on="s")
log(f"baseline held-out {macro_f05_df(pred, tr, ev.select('s')):.5f} (expected 0.99205); eval S1 {ev.height} truth {tr.height}")
print(d.select(FEATS).describe())
