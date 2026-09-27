# Independent estimate: US/India-calibrated probability u(p3, g, pattern) for every France group,
# then the France logit shift each LB group implies, applied to the proposal's removals.
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
u = pl.read_parquet("C:/ber_scratch/france/usi_top.parquet", columns=["label", "p3", "p2g", "p", "pat", "country"])
u = u.with_columns(nk=pl.col("pat").str.split("|").list.get(0), num=pl.col("pat").str.split("|").list.get(1))
print("US/India rows", u.height, "p3==p2g share", (u["p3"] == u["p2g"]).mean())
def lg(x): x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4); return np.log(x / (1 - x))
def X(d):
    nk = d["nk"].to_numpy(); num = d["num"].to_numpy(); a = lg(d["p3"].to_numpy()); c = lg(d["g"].to_numpy())
    cols = [a, c, np.maximum(a, 0), np.maximum(c, 0)]
    for k in ["swap", "added", "other", "dropped", "acronym", "squashed", "empty"]: cols.append((nk == k).astype(float))
    for k in ["ndiff", "nmiss"]: cols.append((num == k).astype(float))
    for k in ["swap", "added"]:
        for m in ["ndiff", "nmiss"]: cols.append(((nk == k) & (num == m)).astype(float))
    return np.column_stack(cols)
u = u.rename({"p2g": "g"})
# only the region of interest to keep the fit local: rescored pairs (p3 != g) and g >= 0.02 or p3 >= 0.02
ur = u.filter(pl.col("p3") != pl.col("g"))
print("US/India rescored best candidates", ur.height, "label rate", ur["label"].mean())
lr = LogisticRegression(C=10, max_iter=2000).fit(X(ur), ur["label"].to_numpy())
ur = ur.with_columns(uhat=lr.predict_proba(X(ur))[:, 1])
print("calibration of US fit by p3 bin (US/India rescored):")
print(ur.with_columns(bin=pl.col("p3").cut([0.2, 0.5, 0.77, 0.9, 0.96, 0.99])).group_by("bin").agg(n=pl.len(), y=pl.col("label").mean(), uhat=pl.col("uhat").mean(), p3=pl.col("p3").mean(), g=pl.col("g").mean()).sort("bin"))
print("US/India pairs like the removals (p3 0.77-0.96, g>=0.8) by pattern:")
print(ur.filter(pl.col("p3").is_between(0.77, 0.96) & (pl.col("g") >= 0.8)).group_by("pat").agg(n=pl.len(), y=pl.col("label").mean(), p3=pl.col("p3").mean(), g=pl.col("g").mean(), p=pl.col("p").mean()).sort("n", descending=True).head(10))
bb = pl.read_parquet("C:/ber_scratch/frfix/refute_lbcal/bb.parquet")
bb = bb.with_columns(uhat=lr.predict_proba(X(bb))[:, 1])
LBt = {"mA": 0.403, "iA": 0.557, "jA": 0.101}
def shift_for(z, t): return brentq(lambda d: (1 / (1 + np.exp(-(z + d)))).mean() - t, -15, 15)
res = {}
for gname in ["mA", "iA", "jA", "gnR", "gnA", "rem", "add"]:
    d = bb.filter(pl.col(gname)); z = lg(d["uhat"].to_numpy())
    line = f"{gname:4s} n {d.height:6d} US-calibrated mean {d['uhat'].mean():.3f}  model pi {d['pi'].mean():.3f}"
    if gname in LBt:
        s = shift_for(z, LBt[gname]); res[gname] = s; line += f"  LB-implied t {LBt[gname]:.3f} -> France logit shift {s:+.2f}"
    print(line)
rem = bb.filter("rem"); zr = lg(rem["uhat"].to_numpy())
for gname, s in res.items():
    print(f"removals' true rate if they carry the {gname} shift {s:+.2f}: {(1 / (1 + np.exp(-(zr + s)))).mean():.3f}")
# gnR constraint: removals of v7g_num. t_r range from LB gn in [-0.002, 0] with gnA at US-shifted rates
for gname, s in res.items():
    ga = bb.filter("gnA"); ta = (1 / (1 + np.exp(-(lg(ga["uhat"].to_numpy()) + s)))).mean()
    for dF in [-0.002, -0.001, 0.0]:
        tr = 0.768 - (dF / 1.156e-6 - ga.height * (ta - 0.768)) / 18250
        print(f"  gn LB {dF:+.3f}, gnA true {ta:.3f} ({gname} shift) -> gnR mean true {tr:.3f}")
bb.select("uhat").write_parquet("C:/ber_scratch/frfix/refute_lbcal/uhat.parquet")
