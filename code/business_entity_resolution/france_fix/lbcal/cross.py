import sys, os, glob; sys.path.insert(0, "C:/ber_scratch/frfix/lbcal")
import numpy as np, time, polars as pl
from model import *
names = [os.path.basename(f)[3:-4] for f in sorted(glob.glob(f"{D}/th_*.npy"))]
TH = {n: np.load(f"{D}/th_{n}.npy") for n in names}
PM = {}
for n, th in TH.items():
    p = pi_of(th); m = miss_of(p, np.exp(th[8])); PM[n] = (p, m)
    o = predict_fast(th)
    print(f"{n:10s} NT {np.exp(th[8]):.0f} " + " ".join(f"{k} {v:+.5f}" for k, v in o.items()) + "  th " + " ".join(f"{x:.2f}" for x in th[:8]))
masks = {}
for n in names:
    p, m = PM[n]; masks["opt_" + n] = decide(p, m)
ens = M["ens"]
FIT = [n for n in names if not n.startswith("loo_")]
allopt = np.stack([masks["opt_" + n] for n in FIT])
cons_rem = ens & ~allopt.any(0)          # removed under every fit
cons_add = ~ens & allopt.all(0)          # added under every fit
masks["consensus"] = (ens & ~cons_rem) | cons_add
maj_rem = ens & ((~allopt).sum(0) > len(FIT) / 2); maj_add = ~ens & (allopt.sum(0) > len(FIT) / 2)
masks["majority"] = (ens & ~maj_rem) | maj_add
masks["cons_remonly"] = ens & ~cons_rem
masks["ens_minus_gnR"] = ens & M["gn"]
masks["v7m"] = M["m"]
print(f"consensus: -{cons_rem.sum()} +{cons_add.sum()}; majority -{maj_rem.sum()} +{maj_add.sum()}")
rows = []
for mk, mask in masks.items():
    r = {"decision": mk, "n": int(mask.sum()), "add": int((mask & ~ens).sum()), "rem": int((~mask & ens).sum())}
    for n in names:
        p, m = PM[n]
        r[n] = round(expF(p, mask, m)[0] - expF(p, ens, m)[0], 5)
    rows.append(r)
d = pl.DataFrame(rows)
d = d.with_columns(minFit=pl.min_horizontal(FIT), minAll=pl.min_horizontal(names), meanAll=pl.mean_horizontal(names))
with pl.Config(tbl_rows=40, tbl_width_chars=250, tbl_cols=30):
    print(d)
np.save(f"{D}/mask_consensus.npy", masks["consensus"]); np.save(f"{D}/mask_majority.npy", masks["majority"]); np.save(f"{D}/mask_cons_remonly.npy", masks["cons_remonly"])
