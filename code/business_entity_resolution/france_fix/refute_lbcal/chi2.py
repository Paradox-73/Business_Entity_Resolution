import sys
sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, glob, os
import model as MD
from model import *
ens = M["ens"]; v3 = b["in_v3"].to_numpy()
add = np.nonzero(v3 & ~ens)[0]; rem = np.nonzero(~v3 & ens)[0]; DIFF["v3"] = (add, rem, np.unique(np.concatenate([sidx[add], sidx[rem]])))
LBX = dict(LB); LBX["v3"] = (-0.01578, 0.0034)
for n in ["full", "tightname", "loo_gn", "loo_m"]:
    th = np.load(f"{D}/th_{n}.npy"); o = predict_fast(th)
    z = {k: (o[k] - v) / s for k, (v, s) in LBX.items()}; z["abs"] = (o["abs"] - 0.8958) / 0.0015; z["E"] = (o["E"] - 0.056) / 0.006
    pr = ((th - TH0) / SIG0) ** 2
    print(n, "LB chi2 (7 pts incl v3)", round(sum(v * v for v in z.values()), 2), "prior chi2", round(pr.sum(), 2), {k: round(v, 2) for k, v in z.items()})
