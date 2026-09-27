import sys, os, glob, time
sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, polars as pl
from model import *
names = [os.path.basename(f)[3:-4] for f in sorted(glob.glob(f"{D}/th_*.npy"))]
v3 = b["in_v3"].to_numpy(); ens = M["ens"]
cm = np.load(f"{D}/mask_consensus.npy"); cr = np.load(f"{D}/mask_cons_remonly.npy")
print("v3 France pairs", v3.sum(), "v3 not ens", (v3 & ~ens).sum(), "ens not v3", (ens & ~v3).sum())
print("LB: v3 - v7ens France = (0.9700-0.838)/0.15 - (0.983159-0.848792)/0.15 =", round((0.9700 - 0.838) / 0.15 - (0.983159 - 0.848792) / 0.15, 5), "+- 0.0034 (v5 given to 3 digits)")
for n in names:
    th = np.load(f"{D}/th_{n}.npy"); p = pi_of(th); m = miss_of(p, np.exp(th[8]))
    fe = expF(p, ens, m)[0]
    out = {k: expF(p, M[k], m)[0] - fe for k in ["m", "i", "j", "gn"]}
    out["v3"] = expF(p, v3, m)[0] - fe
    out["prop"] = expF(p, cm, m)[0] - fe; out["remonly"] = expF(p, cr, m)[0] - fe
    rem = ens & ~cm; add = cm & ~ens
    print(f"{n:11s} " + " ".join(f"{k} {v:+.5f}" for k, v in out.items()) + f" | pi rem {p[rem].mean():.3f} add {p[add].mean():.3f} v3add {p[v3 & ~ens].mean():.3f} v3rem {p[ens & ~v3].mean():.3f}")
