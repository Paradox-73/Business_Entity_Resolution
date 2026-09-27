# Adversarial fits: can the analyst's own model family (and a mildly richer one) fit ALL LB points
# while predicting the proposal is <= 0 ?  If yes, the LB does not pin the sign of the proposal.
import sys, os, time
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lbcal"))
import numpy as np
from scipy.optimize import least_squares
import model as MD
from model import *
OUT = f"{SCRATCH}/frfix/refute_lbcal"
ens = M["ens"]; v3 = b["in_v3"].to_numpy()
cm = np.load(f"{D}/mask_consensus.npy"); cr = np.load(f"{D}/mask_cons_remonly.npy")
for k, mk in [("v3", v3), ("prop", cm), ("remonly", cr)]:
    add = np.nonzero(mk & ~ens)[0]; rem = np.nonzero(~mk & ens)[0]
    DIFF[k] = (add, rem, np.unique(np.concatenate([sidx[add], sidx[rem]])))
p3 = b["p3"].to_numpy(); g = b["g"].to_numpy()
# extra (natural) features: hinge on logit p3 above logit(0.9) and logit(0.99): lets calibration steepen at the top
EXTRA = {
    "h90": np.maximum(LP3 - np.log(9), 0.0),
    "h99": np.maximum(LP3 - np.log(99), 0.0),
    "gh90": np.maximum(LG - np.log(9), 0.0),
}
mode = sys.argv[1]; target = float(sys.argv[2]); tsig = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0002
use_extra = mode == "extra"
XK = list(EXTRA) if use_extra else []
def pi_x(th):
    z = th[0] + th[1] * LP3 + th[2] * LG + th[3] * F["ndiff"] + th[4] * F["nmiss"] + th[5] * F["swap"] + th[6] * F["added"] + th[7] * F["other"]
    for i, k in enumerate(XK): z = z + th[9 + i] * EXTRA[k]
    z = np.where(VETO, z - 3.0, z)
    p = 1 / (1 + np.exp(-z))
    tot = np.bincount(qidx, p, NQ)
    return p * np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)[qidx]
MD.pi_of = pi_x   # predict_fast looks up pi_of in module globals
LBX = dict(LB); LBX["v3"] = (-0.01578, 0.0034)
TH0x = np.concatenate([TH0, np.zeros(len(XK))]); SIG0x = np.concatenate([SIG0, np.full(len(XK), 1.0)])
def res(th):
    o = MD.predict_fast(th)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    r += [(o[k] - v) / s for k, (v, s) in LBX.items()]
    r.append((o["prop"] - target) / tsig)
    r += list((th - TH0x) / SIG0x)
    return np.array(r)
th_full = np.load(f"{D}/th_full.npy")
start = np.concatenate([th_full, np.zeros(len(XK))])
t = time.time()
r = least_squares(res, start, diff_step=1e-3, max_nfev=25, ftol=1e-6, xtol=1e-6)
o = MD.predict_fast(r.x)
p = pi_x(r.x)
rm = ens & ~cm; ad = cm & ~ens
print(f"[{mode} target {target}] {time.time()-t:.0f}s cost {r.cost:.3f} th {np.round(r.x, 3).tolist()} NT {np.exp(r.x[8]):.0f}")
print("   pred", {k: round(float(v), 5) for k, v in o.items()})
print("   LB z-scores", {k: round(float((o[k] - v) / s), 2) for k, (v, s) in LBX.items()}, "abs z", round((o['abs'] - 0.8958) / 0.0015, 2), "E z", round((o['E'] - 0.056) / 0.006, 2))
print(f"   pi removed {p[rm].mean():.3f} added {p[ad].mean():.3f}; v7m add {p[DIFF['m'][0]].mean():.3f} v7i add {p[DIFF['i'][0]].mean():.3f} v7j add {p[DIFF['j'][0]].mean():.3f} gnR {p[DIFF['gn'][1]].mean():.3f} gnA {p[DIFF['gn'][0]].mean():.3f}; ens TP {p[ens].sum():.0f}")
for x in [(0.95, 0.95), (0.86, 0.92), (0.99, 0.99), (0.999, 0.999)]:
    pass
np.save(f"{OUT}/th_adv_{mode}_{target}.npy", r.x)
