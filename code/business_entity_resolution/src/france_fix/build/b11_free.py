"""Class-informed model with the D/X and G/N class true rates as FREE parameters (weak prior, logit 0 +- 3), fitted to
the LB with the v7g_num France value at -0.0022 +- 0.0010 (refuter's value) and at -0.001 +- 0.0007 (lbcal's)."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import numpy as np, time
from scipy.optimize import least_squares
exec(open("b10_classmodel.py").read().split("th_full = np.load")[0])
th_full = np.load(f"{SCRATCH}/frfix/lbcal/th_full.npy")
def pfun2(th):
    p = pi_of(th[:9]); pD, pGN = 1 / (1 + np.exp(-th[9])), 1 / (1 + np.exp(-th[10]))
    p = np.where(Dgrp, pD, np.where(GNgrp, pGN, p))
    tot = np.bincount(qidx, p, NQ); sc = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
    return p * sc[qidx]
pfun_orig = pfun
def resid2(th, gn):
    global pfun
    o = pred2(th)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    for k, (v, s) in LB.items():
        if k == "gn": v, s = gn
        r.append((o[k] - v) / s)
    r += list((th[:9] - TH0) / SIG0) + [th[9] / 3, th[10] / 3]
    return np.array(r)
def pred2(th):
    import builtins
    g = globals(); g["pfun"] = lambda t: pfun2(th)
    o = pred(th[:9]); g["pfun"] = pfun_orig
    return o
for gn in [(-0.0022, 0.0010), (-0.001, 0.0007)]:
    for start in [(-3.5, 3.2), (-1.0, 1.0)]:
        t = time.time()
        x0 = np.concatenate([th_full, start])
        res = least_squares(resid2, x0, diff_step=1e-3, max_nfev=40, ftol=1e-6, xtol=1e-6, kwargs=dict(gn=gn))
        o = pred2(res.x); rr = resid2(res.x, gn)[:6]
        pD, pGN = 1 / (1 + np.exp(-res.x[9])), 1 / (1 + np.exp(-res.x[10]))
        print(f"[gn={gn} start={start}] {time.time()-t:.0f}s LB chi2 {np.sum(rr**2):.2f} pD={pD:.3f} pGN={pGN:.3f} th {np.round(res.x[:9], 2).tolist()}")
        print("    ", {k: round(v, 5) for k, v in o.items()}, flush=True)
