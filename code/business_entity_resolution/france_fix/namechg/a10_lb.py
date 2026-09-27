"""Calibrate the sibling statistic on LB-tested France changes (all S1 rows, no descriptor restriction)."""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl, numpy as np
OUT = "C:/ber_scratch/frfix/namechg"; FR = "C:/ber_scratch/france"
a = pl.read_parquet(f"{OUT}/fr_all_cls.parquet")
a = a.with_columns(sameacc=((pl.col("cls") == "samename") & pl.col("acc")))
a = a.with_columns(oth=pl.col("sameacc").sum().over("s") - pl.col("sameacc").cast(pl.Int64),
                   kind=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last())
T = a.filter((pl.col("cls") == "samename") & (pl.col("num") == "nsame") & pl.col("acc"))
Dc = a.filter((pl.col("cls") == "X") & (pl.col("num") == "ndiff") | ((pl.col("cls") == "G") & (pl.col("kind") == "added") & (pl.col("num") == "ndiff")))
def st(x):
    o = x["oth"].to_numpy().astype(float); return o.mean(), o.std()/np.sqrt(len(o)), (o == 0).mean(), np.sqrt((o == 0).mean()*(1-(o == 0).mean())/len(o))
tm, _, t0, _ = st(T); dm, _, d0, _ = st(Dc)
print(f"refs all-S1: TRUE oth {tm:.4f} oth0 {t0:.4f} | DECOY oth {dm:.4f} oth0 {d0:.4f}")
def est(x, name):
    m, s, z, zs = st(x)
    tmn, tz = (dm-m)/(dm-tm), (d0-z)/(d0-t0); sm, sz = s/(dm-tm), zs/(d0-t0)
    w = np.array([1/sm**2, 1/sz**2]); comb = (w[0]*tmn+w[1]*tz)/w.sum()
    print(f"{name:52s} n={x.height:6d} true share: mean {tmn:.2f}+-{sm:.2f} zero {tz:.2f}+-{sz:.2f} combined {comb:.2f}+-{1/np.sqrt(w.sum()):.2f}")
E = pl.read_parquet(f"{FR}/pairs_v7ens.parquet")
M = pl.read_parquet(f"{OUT}/pairs_v7m.parquet")
for nm, V in (("v7m", M), ("v7i", pl.read_parquet(f"{FR}/pairs_v7i.parquet")), ("v7j", pl.read_parquet(f"{FR}/pairs_v7j.parquet")),
              ("v7g_num", pl.read_parquet(f"{FR}/pairs_v7g_num.parquet"))):
    add = V.join(E, on=["q", "s"], how="anti").join(a, on=["q", "s"])
    rem = E.join(V, on=["q", "s"], how="anti").join(a, on=["q", "s"])
    if add.height > 1000: est(add, f"{nm} ADDED pairs")
    if rem.height > 1000: est(rem, f"{nm} REMOVED pairs")
    if nm == "v7m":
        for c in ("D", "G", "N"):
            est(add.filter(pl.col("cls") == c), f"   v7m added, class {c}")
        est(add.filter(~pl.col("cls").is_in(["D", "G", "N"])), f"   v7m added, other classes")
    if nm == "v7g_num":
        est(rem.filter(pl.col("num") == "ndiff"), "   v7g_num removed number-differing")
        est(rem.filter(pl.col("num") != "ndiff"), "   v7g_num removed others")
x = a.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame") & pl.col("acc"))
est(x, "PROPOSED VETO: D-class same-number accepted")
est(a.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame")), "D-class same-number, all statuses")
