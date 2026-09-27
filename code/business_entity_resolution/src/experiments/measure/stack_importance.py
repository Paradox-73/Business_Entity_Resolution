"""Methodology section 4 ("What it relies on"): split gain of each feature of the v10d stacker, as a share of each
model's total gain, averaged over the 4 LightGBM models of sets/lgb_models_avg.pkl.

  python stack_importance.py        (from any folder; prints; BER_SETS overrides the sets folder)
"""
import os
import pickle

import numpy as np

SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sets"))
m = pickle.load(open(os.path.join(SETS, "lgb_models_avg.pkl"), "rb"))
print(type(m), list(m.keys()))
names = m[list(m.keys())[0]].feature_name()
print(len(names), names)
G = np.zeros(len(names))
S = np.zeros(len(names))
for k, b in m.items():
    g = b.feature_importance("gain")
    s = b.feature_importance("split")
    G += g / g.sum()
    S += s / s.sum()
    print(k, b.num_trees(), b.params.get("num_leaves") if b.params else None)
G /= len(m)
S /= len(m)
for i in np.argsort(-G):
    print(f"{names[i]:10s} gain {G[i] * 100:6.2f}%  split {S[i] * 100:6.2f}%")
