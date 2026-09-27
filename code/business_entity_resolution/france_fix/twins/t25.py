# within-subset permutation null (the replacement word drawn from the same subset's word distribution)
import polars as pl, numpy as np
from collections import Counter
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
top = pl.read_parquet(f"{T}/top_k3.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyc = set(zip(s1["ks"].to_list(), s1["scity"].to_list()))
keyn = set(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["snum"].fill_null("-").to_list()))
keycf = set(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["fs"].to_list()))
f = top.filter(pl.col("pat") == "swap|nsame").with_columns(p3b=pl.col("p3").cut([0.1, 0.5, 0.9, 0.99]).cast(pl.Utf8))
for sub in ["acc_true", "acc_false", "p3b"]:
    groups = [("accepted", f.filter(pl.col("acc"))), ("unaccepted", f.filter(~pl.col("acc")))] if sub.startswith("acc") else [(k[0], g) for k, g in f.group_by("p3b")]
    if sub == "acc_false": continue
    for name, g in sorted(groups, key=lambda x: x[0]):
        kq = g["kq"].to_list(); ks = g["ks"].to_list(); city = g["qcity"].to_list(); num = g["qnum"].fill_null("-").to_list(); fq = g["fq"].to_list()
        shared, new = [], []
        for a, b in zip(kq, ks):
            A, B = set(a.split()), set(b.split()); shared.append(A & B); new.append(sorted(A - B))
        oc = np.array([(k, c) in keyc for k, c in zip(kq, city)]).sum()
        on = np.array([(k, c, n) in keyn for k, c, n in zip(kq, city, num)]).sum()
        of = np.array([(k, c, x) in keycf for k, c, x in zip(kq, city, fq)]).sum()
        rng = np.random.default_rng(3); sc = []; sn = []; sf = []
        for rep in range(4):
            perm = rng.permutation(len(new))
            ks2 = [" ".join(sorted(shared[i] | set(new[j]))) for i, j in enumerate(perm)]
            sc.append(sum((k, c) in keyc and k != o for k, c, o in zip(ks2, city, ks)))
            sn.append(sum((k, c, n) in keyn and k != o for k, c, n, o in zip(ks2, city, num, ks)))
            sf.append(sum((k, c, x) in keycf and k != o for k, c, x, o in zip(ks2, city, fq, ks)))
        print(f"swap|nsame {name:12s} n={len(kq):6d} | city twin obs {oc:6d} null {np.mean(sc):8.1f} | same-num twin obs {on:5d} null {np.mean(sn):7.1f} | same-form city twin obs {of:5d} null {np.mean(sf):7.1f}")
