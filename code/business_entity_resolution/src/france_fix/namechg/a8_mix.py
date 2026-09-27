"""Mixture estimate of true share per group from the 'other accepted same-name records of the S1' statistic.
References: TRUE = same-name same-number accepted; DECOY = pure-decoy word (X) or G-added with different number.
Restricted to S1 rows whose name has a descriptor (D-class) word, so all groups come from comparable S1 rows."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
import numpy as np
import france_cal as fc
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
OUT = f"{SCRATCH}/frfix/namechg"
FR = f"{SCRATCH}/france"
wc = pl.read_parquet(f"{OUT}/word_class.parquet")
D = set(wc.filter(pl.col("cls") == "D")["a"].to_list())
a = pl.read_parquet(f"{OUT}/fr_all_cls.parquet")
top = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "sn"])
a = a.join(top, on="q")
a = a.with_columns(sameacc=((pl.col("cls") == "samename") & pl.col("acc")))
a = a.with_columns(oth=pl.col("sameacc").sum().over("s") - pl.col("sameacc").cast(pl.Int64),
                   allrec=pl.len().over("s") - 1)
sd = a.select("s", "sn").unique("s")
sd = sd.with_columns(hasD=pl.Series([any(w in D for w in fc.toks(x)) for x in sd["sn"].to_list()]))
a = a.join(sd.select("s", "hasD"), on="s").filter(pl.col("hasD"))
a = a.with_columns(kind=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last(),
                   rest=pl.col("in7m") & ~pl.col("acc"))
a = a.with_columns(grp=pl.when(pl.col("acc")).then(pl.lit("acc")).when(pl.col("rest")).then(pl.lit("rest")).otherwise(pl.lit("rej")))
a.select("q", "s", "oth", "hasD").write_parquet(f"{OUT}/fr_oth.parquet")
T = a.filter((pl.col("cls") == "samename") & (pl.col("num") == "nsame") & pl.col("acc"))
Dc = a.filter(pl.col("cls").is_in(["X"]) & (pl.col("num") == "ndiff") | ((pl.col("cls") == "G") & (pl.col("kind") == "added") & (pl.col("num") == "ndiff")))


def stats(x):
    o = x["oth"].to_numpy().astype(float)
    return o.mean(), o.std() / np.sqrt(len(o)), (o == 0).mean(), np.sqrt((o == 0).mean() * (1 - (o == 0).mean()) / len(o)), (o <= 1).mean()


tm, ts, t0, t0s, t1 = stats(T)
dm, ds, d0, d0s, d1 = stats(Dc)
print(f"TRUE ref n={T.height} oth {tm:.4f}+-{ts:.4f} oth0 {t0:.4f} oth<=1 {t1:.4f}; DECOY ref n={Dc.height} oth {dm:.4f}+-{ds:.4f} oth0 {d0:.4f} oth<=1 {d1:.4f}")


def est(x, name):
    m, s, z, zs, o1 = stats(x)
    t_mean = (dm - m) / (dm - tm); t_zero = (d0 - z) / (d0 - t0); t_one = (d1 - o1) / (d1 - t1)
    se_m = s / (dm - tm); se_z = zs / (d0 - t0)
    w = np.array([1 / se_m ** 2, 1 / se_z ** 2]); comb = (w[0] * t_mean + w[1] * t_zero) / w.sum()
    print(f"{name:45s} n={x.height:6d} oth {m:.3f} oth0 {z:.4f} -> true share by mean {t_mean:.2f}+-{se_m:.2f}, by zero {t_zero:.2f}+-{se_z:.2f}, by <=1 {t_one:.2f}; combined {comb:.2f}+-{1/np.sqrt(w.sum()):.2f}")
    return comb


for cls in ("D", "G", "N", "rare", "drop"):
    for kind in ("swap", "added", "dropped", "other"):
        for grp in ("acc", "rest", "rej"):
            x = a.filter((pl.col("cls") == cls) & (pl.col("kind") == kind) & (pl.col("num") == "nsame") & (pl.col("grp") == grp))
            if x.height >= 1500:
                est(x, f"{cls} {kind} nsame {grp}")
print("--- D swap nsame accepted by p3 band")
x = a.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame") & pl.col("acc"))
for lo, hi in ((0, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)):
    est(x.filter((pl.col("p3") >= lo) & (pl.col("p3") < hi)), f"D nsame acc p3 [{lo},{hi})")
print("--- D nsame accepted by g band")
for lo, hi in ((0, 0.9), (0.9, 0.99), (0.99, 1.01)):
    est(x.filter((pl.col("p2g") >= lo) & (pl.col("p2g") < hi)), f"D nsame acc g [{lo},{hi})")
print("--- all nsame D (any status)")
est(a.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame")), "D nsame all")
est(a.filter((pl.col("cls") == "D") & (pl.col("num") == "ndiff")), "D ndiff all")
est(a.filter((pl.col("cls") == "samename") & (pl.col("num") == "ndiff") & ~pl.col("acc")), "samename ndiff rejected")
est(a.filter((pl.col("cls") == "samename") & (pl.col("num") == "ndiff") & pl.col("acc")), "samename ndiff accepted")
est(a.filter((pl.col("cls") == "samename") & (pl.col("num") == "nmiss") & pl.col("acc")), "samename nmiss accepted")
est(a.filter((pl.col("cls") == "X") & (pl.col("num") == "nsame")), "X nsame all")
