# twin with SAME house number (different street) in same city: observed vs synthetic
import polars as pl, numpy as np
from collections import Counter
T = r"C:/ber_scratch/frfix/twins"
top = pl.read_parquet(f"{T}/top_k3.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyn = set(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["snum"].fill_null("-").to_list()))
keynf = set(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["snum"].fill_null("-").to_list(), s1["fs"].to_list()))
for PAT in ["swap|nsame", "swap|nmiss", "other|nsame", "added|nsame"]:
    f = top.filter(pl.col("pat") == PAT)
    kq = f["kq"].to_list(); ks = f["ks"].to_list(); city = f["qcity"].to_list(); num = f["qnum"].fill_null("-").to_list(); fq = f["fq"].to_list()
    shared, new = [], []
    for a, b in zip(kq, ks):
        A, B = set(a.split()), set(b.split()); shared.append(A & B); new.append(sorted(A - B))
    obs = np.array([(k, c, n) in keyn for k, c, n in zip(kq, city, num)])
    obsf = np.array([(k, c, n, x) in keynf for k, c, n, x in zip(kq, city, num, fq)])
    rng = np.random.default_rng(2); S = []; SF = []
    for rep in range(4):
        perm = rng.permutation(len(new))
        ks2 = [" ".join(sorted(shared[i] | set(new[j]))) for i, j in enumerate(perm)]
        S.append(np.array([(k, c, n) in keyn and k != o for k, c, n, o in zip(ks2, city, num, ks)]))
        SF.append(np.array([(k, c, n, x) in keynf and k != o for k, c, n, x, o in zip(ks2, city, num, fq, ks)]))
    acc = f["acc"].to_numpy()
    for name, sel in [("all", np.ones(len(kq), bool)), ("accepted", acc), ("unaccepted", ~acc)]:
        print(f"{PAT:12s} {name:10s} n={sel.sum():6d} same-number twin: obs {obs[sel].sum():5d} syn {np.mean([z[sel].sum() for z in S]):7.1f} | same forms: obs {obsf[sel].sum():5d} syn {np.mean([z[sel].sum() for z in SF]):7.1f}")
    f = f.with_columns(twin_num=pl.Series(obs), twin_numf=pl.Series(obsf))
    if PAT == "swap|nsame":
        f.select("q", "s", "twin_num", "twin_numf").write_parquet(f"{T}/twin_num_swap.parquet")
