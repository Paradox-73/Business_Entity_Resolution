import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, time, polars as pl
from model import *
name = sys.argv[1] if len(sys.argv) > 1 else "full"
th = np.load(f"{D}/th_{name}.npy")
p = pi_of(th); m = miss_of(p, np.exp(th[8]))
print("sum pi", p.sum().round(), "miss", m.sum().round())
for k in ["m", "i", "j", "gn"]:
    add, rem, _ = DIFF[k]
    print(f"v7{k}: added {len(add)} mean pi {p[add].mean() if len(add) else 0:.3f} | removed {len(rem)} mean pi {p[rem].mean() if len(rem) else 0:.3f}")
ens = M["ens"]
print(f"v7ens accepted mean pi {p[ens].mean():.4f}  (TP {p[ens].sum():.0f}); rejected candidate mass {p[~ens].sum():.0f}")
t = time.time()
mask = decide(p, m)
print("decide", time.time() - t)
fe = expF(p, ens, m)[0]; fd = expF(p, mask, m)[0]; fm = expF(p, M["m"], m)[0]
add = mask & ~ens; rem = ~mask & ens
print(f"pred F: v7ens {fe:.5f}  v7m {fm:.5f} ({fm-fe:+.5f})  optimal {fd:.5f} ({fd-fe:+.5f}); pairs {mask.sum()} (+{add.sum()} / -{rem.sum()}); mean pi added {p[add].mean():.3f} removed {p[rem].mean():.3f}")
bb = b.select("pat", "g", "p3").with_columns(pi=p, add=add, rem=rem)
print(bb.filter(pl.col("add")).group_by("pat").agg(n=pl.len(), pi=pl.col("pi").mean(), g=pl.col("g").mean(), p3=pl.col("p3").mean()).sort("n", descending=True).head(8))
print(bb.filter(pl.col("rem")).group_by("pat").agg(n=pl.len(), pi=pl.col("pi").mean(), g=pl.col("g").mean(), p3=pl.col("p3").mean()).sort("n", descending=True).head(8))
# decide with ens as base mask: simple decide_expf-like threshold view: calibrated p of v7ens-accepted pairs
np.save(f"{D}/mask_{name}.npy", mask)
