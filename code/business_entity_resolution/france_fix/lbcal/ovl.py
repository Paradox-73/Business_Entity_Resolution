import sys; sys.path.insert(0, "C:/ber_scratch/frfix/lbcal")
import numpy as np
from model import *
th = np.load(f"{D}/th_full.npy"); p = pi_of(th); m = miss_of(p, np.exp(th[8]))
mask = np.load(f"{D}/mask_full.npy"); ens = M["ens"]
rem = ens & ~mask; add = mask & ~ens
gnR = ens & ~M["gn"]; gnA = M["gn"] & ~ens
print("opt removed", rem.sum(), "of which in v7gn removed", (rem & gnR).sum(), "; v7gn removed", gnR.sum())
print("opt added", add.sum(), "in v7i added", (add & M["i"] & ~ens).sum(), "in v7m added", (add & M["m"]).sum(), "in v7gn added", (add & gnA).sum(), "in v7j added", (add & M["j"]).sum())
fe = expF(p, ens, m)[0]
def g(mk): return expF(p, mk, m)[0] - fe
print(f"model: remove gnR only {g(ens & ~gnR):+.5f}; add gnA only {g(ens | gnA):+.5f}; v7gn {g(M['gn']):+.5f}")
print(f"model: opt removals only {g(ens & ~rem):+.5f}; opt additions only {g(ens | add):+.5f}; both {g(mask):+.5f}")
print(f"model: opt removals within gnR {g(ens & ~(rem & gnR)):+.5f}; outside gnR {g(ens & ~(rem & ~gnR)):+.5f}")
print(f"pi of opt-removed in gnR {p[rem & gnR].mean():.3f}, outside {p[rem & ~gnR].mean():.3f}; pi of gnR not removed by opt {p[gnR & ~rem].mean():.3f}")
