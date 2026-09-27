# Alternative model family fitted to the same LB points: France truth = US/India-calibrated probability
# (logistic model trained on 204k labelled US/India pairs) with France corrections:
# logit t = a + b*logit(u_US) + c*(logit g - logit p3) + d_ndiff + d_nmiss (+ veto -3)
import sys, os, time
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lbcal"))
import numpy as np, polars as pl
from scipy.optimize import least_squares
import model as MD
from model import *
OUT = f"{SCRATCH}/frfix/refute_lbcal"
ens = M["ens"]; v3 = b["in_v3"].to_numpy()
cm = np.load(f"{D}/mask_consensus.npy"); cr = np.load(f"{D}/mask_cons_remonly.npy")
for k, mk in [("v3", v3), ("prop", cm), ("remonly", cr)]:
    add = np.nonzero(mk & ~ens)[0]; rem = np.nonzero(~mk & ens)[0]
    DIFF[k] = (add, rem, np.unique(np.concatenate([sidx[add], sidx[rem]])))
U = lg(pl.read_parquet(f"{OUT}/uhat.parquet")["uhat"].to_numpy())
DG = LG - LP3
use_swap = len(sys.argv) > 1 and sys.argv[1] == "swap"
def pi_u(th):
    z = th[0] + th[1] * U + th[2] * DG + th[3] * F["ndiff"] + th[4] * F["nmiss"]
    if use_swap: z = z + th[6] * F["swap"] + th[7] * F["added"]
    z = np.where(VETO, z - 3.0, z)
    p = 1 / (1 + np.exp(-z))
    tot = np.bincount(qidx, p, NQ)
    return p * np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)[qidx]
MD.pi_of = pi_u
LBX = dict(LB); LBX["v3"] = (-0.01578, 0.0034)
# param vector keeps logNT at index 5 via a wrapper
def wrap(x):
    th = np.zeros(9); th[:5] = x[:5]; th[8] = x[5]
    if use_swap: th[6], th[7] = x[6], x[7]
    return th
TH0u = np.array([0, 1, 0, 0, 0, np.log(897.7e3)] + ([0, 0] if use_swap else [])); SIGu = np.array([3, 1, 1, 3, 3, 0.04] + ([3, 3] if use_swap else []))
def res(x):
    o = MD.predict_fast(wrap(x))
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    r += [(o[k] - v) / s for k, (v, s) in LBX.items()]
    r += list((x - TH0u) / SIGu)
    return np.array(r)
t = time.time()
r = least_squares(res, TH0u.copy(), diff_step=1e-3, max_nfev=40, ftol=1e-6, xtol=1e-6)
th = wrap(r.x); o = MD.predict_fast(th); p = pi_u(th)
rm = ens & ~cm; ad = cm & ~ens
print(f"[usfam swap={use_swap}] {time.time()-t:.0f}s cost {r.cost:.3f} x {np.round(r.x, 3).tolist()} NT {np.exp(r.x[5]):.0f}")
print("   pred", {k: round(float(v), 5) for k, v in o.items()})
print("   LB z", {k: round(float((o[k] - v) / s), 2) for k, (v, s) in LBX.items()}, "abs z", round((o['abs'] - 0.8958) / 0.0015, 2), "E z", round((o['E'] - 0.056) / 0.006, 2))
print(f"   pi removed {p[rm].mean():.3f} added {p[ad].mean():.3f}; v7m add {p[DIFF['m'][0]].mean():.3f} v7i add {p[DIFF['i'][0]].mean():.3f} v7j add {p[DIFF['j'][0]].mean():.3f} gnR {p[DIFF['gn'][1]].mean():.3f} gnA {p[DIFF['gn'][0]].mean():.3f}; ens TP {p[ens].sum():.0f}")
bb = b.select("pat").with_columns(pi=p, rem=rm)
print(bb.filter("rem").group_by("pat").agg(n=pl.len(), pi=pl.col("pi").mean()).sort("n", descending=True).head(8))
np.save(f"{OUT}/th_usfam_{int(use_swap)}.npy", th)
