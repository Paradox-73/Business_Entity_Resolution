# Monte Carlo check of the expected-F formula for the PROPOSAL masks (analyst checked only versions)
import sys
sys.path.insert(0, "C:/ber_scratch/frfix/lbcal")
import numpy as np, time
from model import *
th = np.load(f"{D}/th_{sys.argv[1]}.npy")
p = pi_of(th); m = miss_of(p, np.exp(th[8]))
masks = {"ens": M["ens"], "prop": np.load(f"{D}/mask_consensus.npy"), "remonly": np.load(f"{D}/mask_cons_remonly.npy"), "gn": M["gn"], "m": M["m"]}
o = np.argsort(qidx, kind="stable"); qo = qidx[o]; po = p[o]
cum = np.cumsum(po); start = np.ones(len(o), bool); start[1:] = qo[1:] != qo[:-1]
st = np.nonzero(start)[0]; grp = np.cumsum(start) - 1
base = np.concatenate([[0], cum])[st][grp]; hi = cum - base; lo = hi - po
rng = np.random.default_rng(1); res = {k: [] for k in masks}
for it in range(40):
    u = rng.random(NQ)[qo]
    tr = np.zeros(len(p), bool); tr[o] = (u >= lo) & (u < hi)
    T = np.bincount(sidx, tr.astype(float), NSC) + rng.poisson(m)
    for k, mk in masks.items():
        P = np.bincount(sidx, mk.astype(float), NSC); TP = np.bincount(sidx, (mk & tr).astype(float), NSC)
        den = P + 0.25 * T
        f = np.where(den > 0, 1.25 * TP / np.maximum(den, 1e-9), 1.0)
        res[k].append((f.sum() + NS1 - NSC) / NS1)
e = np.array(res["ens"])
for k in masks:
    d = np.array(res[k]) - e
    print(f"{sys.argv[1]} {k}: MC {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}   formula {expF(p, masks[k], m)[0] - expF(p, M['ens'], m)[0]:+.5f}")
