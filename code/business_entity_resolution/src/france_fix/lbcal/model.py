import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import numpy as np, polars as pl
D = f"{SCRATCH}/frfix/lbcal"
NS1 = 259452
b = pl.read_parquet(f"{D}/pairs.parquet")
b = b.with_columns(nk=pl.col("pat").str.split("|").list.get(0), num=pl.col("pat").str.split("|").list.get(1))
sidx = b.select(pl.col("s").rank("dense") - 1)["s"].to_numpy().astype(np.int64)
qidx = b.select(pl.col("q").rank("dense") - 1)["q"].to_numpy().astype(np.int64)
NSC = sidx.max() + 1   # S1 rows with at least one candidate; the rest (NS1-NSC) have no candidates
NQ = qidx.max() + 1
def lg(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))
LP3 = lg(b["p3"].to_numpy()); LG = lg(b["g"].to_numpy())
nk = b["nk"].to_numpy(); num = b["num"].to_numpy()
F = {
    "ndiff": (num == "ndiff").astype(float), "nmiss": (num == "nmiss").astype(float),
    "swap": (nk == "swap").astype(float), "added": (nk == "added").astype(float), "other": (nk == "other").astype(float),
    "street": b["samestreet"].to_numpy().astype(float),
}
VETO = b["veto"].to_numpy()
M = {k: b["in_" + k].to_numpy() for k in ["ens", "m", "i", "j", "gn"]}
PARAMS = ["a", "b", "c", "ndiff", "nmiss", "swap", "added", "other", "logNT"]

def pi_of(th):
    a, bb, c = th[0], th[1], th[2]
    z = a + bb * LP3 + c * LG + th[3] * F["ndiff"] + th[4] * F["nmiss"] + th[5] * F["swap"] + th[6] * F["added"] + th[7] * F["other"]
    z = np.where(VETO, z - 3.0, z)       # legal-form veto: US conflicts are 5% true
    p = 1 / (1 + np.exp(-z))
    tot = np.bincount(qidx, p, NQ)
    scale = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
    return p * scale[qidx]

def miss_of(p, NT):
    w = np.bincount(sidx, p, NSC) + 0.1
    mt = max(NT - p.sum(), 0.0)
    return mt * w / w.sum()

def expF(p, mask, m):
    """Expected macro F0.5 over ALL France S1 rows (rows without candidates: truth assumed empty -> 1)."""
    pm = np.where(mask, p, 0.0); pr = np.where(mask, 0.0, p)
    k = np.bincount(sidx, mask.astype(float), NSC)
    x = np.bincount(sidx, pm, NSC); vx = np.bincount(sidx, pm * (1 - pm), NSC)
    r = np.bincount(sidx, pr, NSC) + m; vr = np.bincount(sidx, pr * (1 - pr), NSC) + m
    p0 = np.exp(np.bincount(sidx, np.log(np.clip(1 - p, 1e-12, 1)), NSC) - m)
    Dn = k + 0.25 * x + 0.25 * r
    Dn1 = np.maximum(Dn, 1e-9)
    f = 1.25 * x / Dn1 - 0.3125 * (k + 0.25 * r) / Dn1 ** 3 * vx + 0.078125 * x / Dn1 ** 3 * vr
    f = np.where(k > 0, np.maximum(f, 0), p0)
    E = (p0.sum() + (NS1 - NSC)) / NS1
    return (f.sum() + (NS1 - NSC)) / NS1, E

LB = {"m": (-0.0091, 0.0004), "i": (-0.0035, 0.0004), "j": (-0.0163, 0.0004), "gn": (-0.001, 0.0007)}
TH0 = np.array([0, 0.5, 0.5, 0, 0, 0, 0, 0, np.log(897.7e3)])
SIG0 = np.array([3, 1.0, 1.0, 3, 3, 3, 3, 3, 0.04])

def predict(th):
    p = pi_of(th); m = miss_of(p, np.exp(th[8]))
    fe, E = expF(p, M["ens"], m)
    out = {"ens": fe, "E": E, "abs": fe - E}
    for k in LB: out[k] = expF(p, M[k], m)[0] - fe
    return out

def resid(th, drop=(), fixed=None, prior_w=1.0):
    o = predict(th)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    for k, (v, s) in LB.items():
        r.append(0.0 if k in drop else (o[k] - v) / s)
    r += list(prior_w * (th - TH0) / SIG0)
    return np.array(r)

def decide(p, m, veto=VETO):
    """Per S1 optimal top-k over records whose best candidate (by p) is this S1; returns bool mask over pairs."""
    pv = np.where(veto, 0.0, p)
    order = np.lexsort((-pv, qidx))           # by q, then p desc
    first = np.ones(len(order), bool); first[1:] = qidx[order][1:] != qidx[order][:-1]
    best = np.zeros(len(p), bool); best[order[first]] = True
    best &= pv > 0.02
    Rt = np.bincount(sidx, p, NSC) + m
    Vt = np.bincount(sidx, p * (1 - p), NSC) + m
    p0 = np.exp(np.bincount(sidx, np.log(np.clip(1 - p, 1e-12, 1)), NSC) - m)
    idx = np.nonzero(best)[0]
    o = idx[np.lexsort((-pv[idx], sidx[idx]))]
    ss = sidx[o]; pp = pv[o]
    start = np.ones(len(o), bool); start[1:] = ss[1:] != ss[:-1]
    grp = np.cumsum(start) - 1
    cs = np.cumsum(pp); cv = np.cumsum(pp * (1 - pp))
    st = np.nonzero(start)[0]
    base_cs = np.concatenate([[0], cs])[st][grp]; base_cv = np.concatenate([[0], cv])[st][grp]
    x = cs - base_cs; vx = cv - base_cv
    kk = np.arange(len(o)) - st[grp] + 1
    r = Rt[ss] - x; vr = np.maximum(Vt[ss] - vx, 0)
    Dn = kk + 0.25 * x + 0.25 * r
    f = 1.25 * x / Dn - 0.3125 * (kk + 0.25 * r) / Dn ** 3 * vx + 0.078125 * x / Dn ** 3 * vr
    # best k per group vs k=0
    fb = np.full(NSC, -1.0); np.maximum.at(fb, ss, f)
    kbest = np.zeros(NSC, np.int64)
    ismax = f >= fb[ss] - 1e-15
    kb = np.full(NSC, 10**9); np.minimum.at(kb, ss, np.where(ismax, kk, 10**9))
    choose = (kk <= kb[ss]) & (fb[ss] > p0[ss])
    mask = np.zeros(len(p), bool); mask[o[choose]] = True
    return mask

# fast version: aggregates of ens once, versions via their differing pairs only
DIFF = {}
for _k in LB:
    add = np.nonzero(M[_k] & ~M["ens"])[0]; rem = np.nonzero(~M[_k] & M["ens"])[0]
    DIFF[_k] = (add, rem, np.unique(np.concatenate([sidx[add], sidx[rem]])))

def _f(k, x, vx, r, vr, p0):
    Dn1 = np.maximum(k + 0.25 * x + 0.25 * r, 1e-9)
    f = 1.25 * x / Dn1 - 0.3125 * (k + 0.25 * r) / Dn1 ** 3 * vx + 0.078125 * x / Dn1 ** 3 * vr
    return np.where(k > 0, np.maximum(f, 0), p0)

def predict_fast(th, masks=None):
    p = pi_of(th); m = miss_of(p, np.exp(th[8]))
    mask = M["ens"]
    pm = np.where(mask, p, 0.0); pr = p - pm
    k = np.bincount(sidx, mask.astype(float), NSC)
    x = np.bincount(sidx, pm, NSC); vx = np.bincount(sidx, pm * (1 - pm), NSC)
    r = np.bincount(sidx, pr, NSC) + m; vr = np.bincount(sidx, pr * (1 - pr), NSC) + m
    p0 = np.exp(np.bincount(sidx, np.log(np.clip(1 - p, 1e-12, 1)), NSC) - m)
    f = _f(k, x, vx, r, vr, p0)
    fe = (f.sum() + (NS1 - NSC)) / NS1; E = (p0.sum() + (NS1 - NSC)) / NS1
    out = {"ens": fe, "E": E, "abs": fe - E}
    for kk, (add, rem, us) in DIFF.items():
        dk = np.zeros(NSC); dx = np.zeros(NSC); dv = np.zeros(NSC)
        np.add.at(dk, sidx[add], 1); np.add.at(dx, sidx[add], p[add]); np.add.at(dv, sidx[add], p[add] * (1 - p[add]))
        np.add.at(dk, sidx[rem], -1); np.add.at(dx, sidx[rem], -p[rem]); np.add.at(dv, sidx[rem], -p[rem] * (1 - p[rem]))
        f2 = _f(k[us] + dk[us], x[us] + dx[us], vx[us] + dv[us], r[us] - dx[us], vr[us] - dv[us], p0[us])
        out[kk] = (f2.sum() - f[us].sum()) / NS1
    return out

def resid_fast(th, drop=(), prior_w=1.0, sig0=None, th0=None):
    o = predict_fast(th)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    for k, (v, s) in LB.items():
        r.append(0.0 if k in drop else (o[k] - v) / s)
    r += list(prior_w * (th - (TH0 if th0 is None else th0)) / (SIG0 if sig0 is None else sig0))
    return np.array(r)
