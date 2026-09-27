import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
from collections import Counter
T = rf"{SCRATCH}/frfix/twins"
top = pl.read_parquet(f"{T}/top_k3.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
cnt = Counter(zip(s1["ks"].to_list(), s1["scity"].to_list()))
cntf = Counter(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["fs"].to_list()))
for PAT in ["swap|nsame", "swap|nmiss", "added|nsame", "other|nsame"]:
    f = top.filter(pl.col("pat") == PAT)
    kq = f["kq"].to_list(); ks = f["ks"].to_list(); city = f["qcity"].to_list(); fq = f["fq"].to_list()
    shared, new = [], []
    for a, b in zip(kq, ks):
        A, B = set(a.split()), set(b.split()); shared.append(A & B); new.append(sorted(A - B))
    obs = np.array([cnt.get((k, c), 0) for k, c in zip(kq, city)])
    obsf = np.array([cntf.get((k, c, x), 0) for k, c, x in zip(kq, city, fq)])
    rng = np.random.default_rng(1)
    S, SF = [], []
    for rep in range(4):
        perm = rng.permutation(len(new))
        ks2 = [" ".join(sorted(shared[i] | set(new[j]))) for i, j in enumerate(perm)]
        S.append(np.array([cnt.get((k, c), 0) if k != o else 0 for k, c, o in zip(ks2, city, ks)]))
        SF.append(np.array([cntf.get((k, c, x), 0) if k != o else 0 for k, c, x, o in zip(ks2, city, fq, ks)]))
    acc = f["acc"].to_numpy()
    print(f"== {PAT}  n={len(kq)} accepted={acc.sum()}")
    for name, sel in [("all", np.ones(len(kq), bool)), ("accepted", acc), ("unaccepted", ~acc)]:
        for lab, o, s in [("any-form", obs, S), ("same-form", obsf, SF)]:
            line = f"  {name:10s} {lab:9s}"
            for bl, fn in [("nx=1", lambda x: x == 1), ("nx=2-3", lambda x: (x >= 2) & (x <= 3)), ("nx>=4", lambda x: x >= 4), ("any", lambda x: x >= 1)]:
                ob = fn(o[sel]).mean(); sy = np.mean([fn(z[sel]).mean() for z in s])
                line += f" | {bl}: obs {ob:.4f} syn {sy:.4f} excess {(ob - sy) * sel.sum():7.0f}"
            print(line)
