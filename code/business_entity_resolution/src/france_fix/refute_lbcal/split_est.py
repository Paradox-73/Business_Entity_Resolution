import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import numpy as np, polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
D = f"{SCRATCH}/frfix/refute_lbcal"
bb = pl.concat([pl.read_parquet(f"{D}/bb.parquet"), pl.read_parquet(f"{D}/uhat.parquet")], how="horizontal")
def lg(x): x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4); return np.log(x / (1 - x))
sg = lambda z: 1 / (1 + np.exp(-z))
# shifts implied by LB groups restricted to each number kind
from scipy.optimize import brentq
def shift(d, t): z = lg(d["uhat"].to_numpy()); return brentq(lambda s: sg(z + s).mean() - t, -15, 15)
print("mA by num:", bb.filter("mA").group_by("num").agg(n=pl.len(), u=pl.col("uhat").mean()).to_dicts())
print("jA by num:", bb.filter("jA").group_by("num").agg(n=pl.len(), u=pl.col("uhat").mean()).to_dicts())
print("iA by num:", bb.filter("iA").group_by("num").agg(n=pl.len(), u=pl.col("uhat").mean(), g=pl.col("g").mean(), p3=pl.col("p3").mean()).to_dicts())
sm = shift(bb.filter("mA"), 0.403); sj = shift(bb.filter("jA"), 0.101); si = shift(bb.filter("iA"), 0.557)
rem = bb.filter("rem").with_columns(z=pl.Series(lg(bb.filter("rem")["uhat"].to_numpy())))
out = rem.group_by("num").agg(n=pl.len(), u=pl.col("uhat").mean(), pi_model=pl.col("pi").mean(),
    t_mA=pl.col("z").map_batches(lambda z: pl.Series(sg(z.to_numpy() + sm))).mean(),
    t_iA=pl.col("z").map_batches(lambda z: pl.Series(sg(z.to_numpy() + si))).mean(),
    t_jA=pl.col("z").map_batches(lambda z: pl.Series(sg(z.to_numpy() + sj))).mean()).sort("num")
print(f"shifts: mA {sm:+.2f} iA {si:+.2f} jA {sj:+.2f}"); print(out)
# micro-approx France delta of the removals under 'nearest analog': nsame->mA, ndiff->jA, nmiss->mean(mA,jA) and ->iA
k = 1.156e-6; be = 0.768
def dF(t, n): return k * n * (be - t)
r = {row["num"]: row for row in out.to_dicts()}
tot_a = dF(r["nsame"]["t_mA"], r["nsame"]["n"]) + dF(r["ndiff"]["t_jA"], r["ndiff"]["n"]) + dF(r["nmiss"]["t_iA"], r["nmiss"]["n"])
tot_b = dF(r["nsame"]["t_mA"], r["nsame"]["n"]) + dF(r["ndiff"]["t_jA"], r["ndiff"]["n"]) + dF(r["nmiss"]["t_mA"], r["nmiss"]["n"])
tot_m = sum(dF(r[x]["pi_model"], r[x]["n"]) for x in r)
print(f"micro France delta of removals: analog (nmiss as iA) {tot_a:+.5f}; analog (nmiss as mA) {tot_b:+.5f}; analyst model pi {tot_m:+.5f}")
# swap|nsame alone
d = rem.filter(pl.col("pat") == "swap|nsame"); t = sg(d["z"].to_numpy() + sm).mean()
print(f"swap|nsame removals n {d.height} US-cal {d['uhat'].mean():.3f} mA-analog {t:.3f} model {d['pi'].mean():.3f} -> micro delta {dF(t, d.height):+.5f} (model {dF(d['pi'].mean(), d.height):+.5f})")
# mA swap|nsame vs removals swap|nsame: p3 / g
for lab in ["mA", "rem"]:
    d = bb.filter(pl.col(lab) & (pl.col("pat") == "swap|nsame"))
    print(lab, "swap|nsame", d.height, "p3", round(d["p3"].mean(), 3), "g", round(d["g"].mean(), 3), "US-cal", round(d["uhat"].mean(), 3), "model pi", round(d["pi"].mean(), 3))
