# More model families fitted to the same 7 LB numbers (abs, E, v7m, v7i, v7j, v7g_num, v3).
#  mono : monotone piecewise-linear calibration in logit p3 and logit g (knots at p=0.5, 0.9, 0.99, slopes >= 0) + pattern dummies
#  usabs: logit t = a + b*logit(u_US) + c*|logit g - logit p3| + d*(logit g - logit p3) + ndiff + nmiss
# argv: family target tsig   (tsig large = unconstrained)
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
fam, target, tsig = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
K = [0.0, np.log(9), np.log(99)]
def seg(x):
    return [np.minimum(x, K[0]), np.clip(x, K[0], K[1]) - K[0], np.clip(x, K[1], K[2]) - K[1], np.maximum(x, K[2]) - K[2]]
if fam == "mono":
    FE = seg(LP3) + seg(LG) + [F["ndiff"], F["nmiss"], F["swap"], F["added"], F["other"]]
    POS = list(range(8))                 # slopes constrained >= 0 via exp
    x0 = np.concatenate([[0.0], np.log(np.full(8, 0.25)), np.zeros(5), [np.log(897.7e3)]])
    sig = np.concatenate([[3], np.full(8, 2.0), np.full(5, 3.0), [0.04]])
else:
    U = lg(pl.read_parquet(f"{OUT}/uhat.parquet")["uhat"].to_numpy()); DG = LG - LP3
    FE = [U, np.abs(DG), DG, F["ndiff"], F["nmiss"]]
    POS = []
    x0 = np.array([0, 1, 0, 0, 0, 0, np.log(897.7e3)]); sig = np.array([3, 1, 1, 1, 3, 3, 0.04])
FE = np.column_stack(FE)
def coefs(x):
    c = x[1:1 + FE.shape[1]].copy()
    for i in POS: c[i] = np.exp(c[i])
    return c
def pi_f(th):
    x = th
    z = x[0] + FE @ coefs(x)
    z = np.where(VETO, z - 3.0, z)
    p = 1 / (1 + np.exp(-z))
    tot = np.bincount(qidx, p, NQ)
    return p * np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)[qidx]
MD.pi_of = pi_f
def pf(x):
    # predict_fast reads th[8] as logNT: pass a vector whose [8] is logNT
    th = np.zeros(max(len(x), 9)); th[:len(x)] = x
    th8 = x[-1]
    MDx = th.copy(); MDx[8] = th8
    return MDx
def predict(x):
    p = pi_f(x); m = miss_of(p, np.exp(x[-1]))
    return p, m
# own fast predictor (same algebra as model.predict_fast) so logNT can sit at the end
def pred(x):
    p, m = predict(x)
    mask = ens
    pm = np.where(mask, p, 0.0); pr = p - pm
    k = np.bincount(sidx, mask.astype(float), NSC)
    xx = np.bincount(sidx, pm, NSC); vx = np.bincount(sidx, pm * (1 - pm), NSC)
    r = np.bincount(sidx, pr, NSC) + m; vr = np.bincount(sidx, pr * (1 - pr), NSC) + m
    p0 = np.exp(np.bincount(sidx, np.log(np.clip(1 - p, 1e-12, 1)), NSC) - m)
    f = MD._f(k, xx, vx, r, vr, p0)
    fe = (f.sum() + (NS1 - NSC)) / NS1; E = (p0.sum() + (NS1 - NSC)) / NS1
    out = {"ens": fe, "E": E, "abs": fe - E}
    for kk, (add, rem, us) in DIFF.items():
        dk = np.zeros(NSC); dx = np.zeros(NSC); dv = np.zeros(NSC)
        np.add.at(dk, sidx[add], 1); np.add.at(dx, sidx[add], p[add]); np.add.at(dv, sidx[add], p[add] * (1 - p[add]))
        np.add.at(dk, sidx[rem], -1); np.add.at(dx, sidx[rem], -p[rem]); np.add.at(dv, sidx[rem], -p[rem] * (1 - p[rem]))
        f2 = MD._f(k[us] + dk[us], xx[us] + dx[us], vx[us] + dv[us], r[us] - dx[us], vr[us] - dv[us], p0[us])
        out[kk] = (f2.sum() - f[us].sum()) / NS1
    return out, p
LBX = dict(LB); LBX["v3"] = (-0.01578, 0.0034)
def res(x):
    o, _ = pred(x)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    r += [(o[k] - v) / s for k, (v, s) in LBX.items()]
    r.append((o["prop"] - target) / tsig)
    r += list((x - x0) / sig)
    return np.array(r)
t = time.time()
rr = least_squares(res, x0.copy(), diff_step=1e-3, max_nfev=40, ftol=1e-6, xtol=1e-6)
o, p = pred(rr.x)
z = {k: round(float((o[k] - v) / s), 2) for k, (v, s) in LBX.items()}; z["abs"] = round((o["abs"] - 0.8958) / 0.0015, 2); z["E"] = round((o["E"] - 0.056) / 0.006, 2)
rm = ens & ~cm; ad = cm & ~ens
print(f"[{fam} target {target} tsig {tsig}] {time.time()-t:.0f}s cost {rr.cost:.3f} coefs a={rr.x[0]:.3f} {np.round(coefs(rr.x), 3).tolist()} NT {np.exp(rr.x[-1]):.0f}")
print("   pred", {k: round(float(v), 5) for k, v in o.items()})
print("   LB z", z, "LB chi2", round(sum(v * v for v in z.values()), 2))
print(f"   pi removed {p[rm].mean():.3f} added {p[ad].mean():.3f}; v7m add {p[DIFF['m'][0]].mean():.3f} v7i add {p[DIFF['i'][0]].mean():.3f} v7j add {p[DIFF['j'][0]].mean():.3f} gnR {p[DIFF['gn'][1]].mean():.3f}; ens TP {p[ens].sum():.0f}")
np.save(f"{OUT}/x_{fam}_{target}_{tsig}.npy", rr.x)
