import sys; sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, time
from model import *
name = sys.argv[1] if len(sys.argv) > 1 else "full"
th = np.load(f"{D}/th_{name}.npy")
p = pi_of(th); m = miss_of(p, np.exp(th[8]))
masks = {k: M[k] for k in ["ens", "m", "i", "j", "gn"]}
masks["opt"] = np.load(f"{D}/mask_{name}.npy")
# record-level categorical sampling: order pairs by q
o = np.argsort(qidx, kind="stable"); qo = qidx[o]; po = p[o]
cum = np.cumsum(po); start = np.ones(len(o), bool); start[1:] = qo[1:] != qo[:-1]
st = np.nonzero(start)[0]; grp = np.cumsum(start) - 1
base = np.concatenate([[0], cum])[st][grp]
hi = cum - base; lo = hi - po           # within-record cumulative intervals
rng = np.random.default_rng(0)
res = {k: [] for k in masks}
t = time.time()
for it in range(30):
    u = rng.random(NQ)[qo]
    tr = np.zeros(len(p), bool); tr[o] = (u >= lo) & (u < hi)
    T = np.bincount(sidx, tr.astype(float), NSC) + rng.poisson(m)
    for k, mk in masks.items():
        P = np.bincount(sidx, mk.astype(float), NSC); TP = np.bincount(sidx, (mk & tr).astype(float), NSC)
        den = P + 0.25 * T
        f = np.where(den > 0, 1.25 * TP / np.maximum(den, 1e-9), 1.0)
        res[k].append((f.sum() + NS1 - NSC) / NS1)
print("MC", time.time() - t)
e = np.array(res["ens"])
print(f"ens {e.mean():.5f} +- {e.std()/np.sqrt(len(e)):.5f}")
for k in masks:
    d = np.array(res[k]) - e
    print(f"  {k}: MC diff {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}   delta-method {expF(p, masks[k], m)[0] - expF(p, M['ens'], m)[0]:+.5f}")
print("delta ens", expF(p, M["ens"], m))
