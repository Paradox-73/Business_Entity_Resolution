import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import WORK  # noqa: E402
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from model import *
cm = np.load(f"{D}/mask_consensus.npy"); ens = M["ens"]
thf = np.load(f"{D}/th_full.npy"); tht = np.load(f"{D}/th_tightname.npy")
bb = b.select("q", "s", "pat", "g", "p3", "pe", "samestreet").with_columns(chosen=cm, rem=ens & ~cm, add=cm & ~ens, gnR=ens & ~M["gn"], inI=M["i"] & ~ens,
                                                              pif=pi_of(thf), pit=pi_of(tht))
for lab in ["rem", "add"]:
    d = bb.filter(pl.col(lab))
    print(f"== {lab} {d.height}: mean g {d['g'].mean():.3f} p3 {d['p3'].mean():.3f} pi_full {d['pif'].mean():.3f} pi_tight {d['pit'].mean():.3f}; in v7gn-removed {d['gnR'].sum()}, in v7i-added {d['inI'].sum()}")
    print(d.group_by("pat").agg(n=pl.len(), g=pl.col("g").mean(), p3=pl.col("p3").mean(), pif=pl.col("pif").mean(), pit=pl.col("pit").mean()).sort("n", descending=True).head(8))
    print(d.select(pl.col("p3").quantile(0.1).alias("p3_q10"), pl.col("p3").quantile(0.9).alias("p3_q90"), pl.col("g").quantile(0.1).alias("g_q10"), pl.col("g").quantile(0.9).alias("g_q90")))
# scores file: v7ens production scores, France p2 -> 0.99 chosen / min(p2, 0.3) otherwise
fr = bb.select("q", "s", "chosen")
src = pl.read_parquet(rf"{WORK}/ce/test_scores_blend_ab_a2_frmin.parquet")
out = src.join(fr, on=["q", "s"], how="left").with_columns(
    p2=pl.when(pl.col("chosen") == True).then(0.99).when(pl.col("chosen") == False).then(pl.min_horizontal(pl.col("p2"), pl.lit(0.3))).otherwise(pl.col("p2")).cast(src["p2"].dtype)).drop("chosen")
assert out.height == src.height
print("France rows changed:", (out["p2"] != src["p2"]).sum(), "non-France rows changed:", out.join(fr, on=["q", "s"], how="anti").height - src.join(fr, on=["q", "s"], how="anti").height)
out.write_parquet(f"{D}/test_scores_frlbcal.parquet")
bb.filter(pl.col("chosen")).select("s", "q").write_parquet(f"{D}/pairs_frlbcal.parquet")
print("wrote")
