import sys, os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lbcal"))
import numpy as np, polars as pl
from model import *
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
ens = M["ens"]; cm = np.load(f"{D}/mask_consensus.npy")
thf = np.load(f"{D}/th_full.npy"); pif = pi_of(thf)
bb = b.select("q", "s", "pat", "nk", "num", "g", "p3", "samestreet").with_columns(
    pi=pif, rem=ens & ~cm, add=cm & ~ens, gnR=ens & ~M["gn"], gnA=M["gn"] & ~ens, mA=M["m"] & ~ens, iA=M["i"] & ~ens, jA=M["j"] & ~ens, ens=ens)
print("== gnR split by in-proposal-removal and number kind")
print(bb.filter("gnR").group_by("rem", "num").agg(n=pl.len(), g=pl.col("g").mean(), p3=pl.col("p3").mean(), pi=pl.col("pi").mean(), st=pl.col("samestreet").mean()).sort("rem", "num"))
print("== proposal removals NOT in gnR")
print(bb.filter(pl.col("rem") & ~pl.col("gnR")).group_by("pat").agg(n=pl.len(), g=pl.col("g").mean(), p3=pl.col("p3").mean(), pi=pl.col("pi").mean()).sort("n", descending=True).head(10))
print("== p3 distribution of removals vs all accepted")
for lab, f in [("rem", pl.col("rem")), ("ens", pl.col("ens")), ("gnR", pl.col("gnR")), ("mA", pl.col("mA")), ("iA", pl.col("iA")), ("jA", pl.col("jA"))]:
    d = bb.filter(f)
    print(lab, d.height, "p3 q10/50/90", [round(d["p3"].quantile(x), 3) for x in (0.1, 0.5, 0.9)], "g q10/50/90", [round(d["g"].quantile(x), 3) for x in (0.1, 0.5, 0.9)], "pi", round(d["pi"].mean(), 3), "p3==g share", round((d["p3"] == d["g"]).mean(), 3))
bb.write_parquet(f"{SCRATCH}/frfix/refute_lbcal/bb.parquet")
