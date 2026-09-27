import sys; sys.path.insert(0, "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal")
import numpy as np, time
from model import *
print("NSC", NSC, "NQ", NQ, "pairs", len(sidx))
for k, v in M.items(): print(k, v.sum(), "S1 nonempty", len(np.unique(sidx[v])))
# naive: pi = pens (after veto), N_T = 897.7k
pe = b["pe"].to_numpy().astype(float)
tot = np.bincount(qidx, pe, NQ); pe = pe * np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1)[qidx]
t = time.time()
for NT in [860e3, 897.7e3, 930e3]:
    m = miss_of(pe, NT)
    base = expF(pe, M["ens"], m)
    print(f"pens as truth, NT {NT:.0f}: sum p {pe.sum():.0f} ens F {base[0]:.4f} E {base[1]:.4f} F-E {base[0]-base[1]:.4f}", {k: round(expF(pe, M[k], m)[0] - base[0], 4) for k in ["m", "i", "j", "gn"]})
print(time.time() - t)
