# null model: are twins (record's exact name = another S1 in same city) more frequent than chance?
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
T = rf"{SCRATCH}/frfix/twins"
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(330)
top = pl.read_parquet(f"{T}/top_k3.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyset = set(zip(s1["ks"].to_list(), s1["scity"].to_list()))
keyform = set(zip(s1["ks"].to_list(), s1["scity"].to_list(), s1["fs"].to_list()))
f = top.filter(pl.col("pat") == "swap|nsame")
kq = f["kq"].to_list(); ks = f["ks"].to_list(); city = f["qcity"].to_list(); fq = f["fq"].to_list()
shared, new = [], []
for a, b in zip(kq, ks):
    A, B = set(a.split()), set(b.split())
    shared.append(A & B); new.append(sorted(A - B))
rng = np.random.default_rng(0)
obs = np.array([(k, c) in keyset for k, c in zip(kq, city)])
obsf = np.array([(k, c, x) in keyform for k, c, x in zip(kq, city, fq)])
res = []
for rep in range(3):
    perm = rng.permutation(len(new))
    syn = []
    synf = []
    for i, j in enumerate(perm):
        k = " ".join(sorted(shared[i] | set(new[j])))
        syn.append((k, city[i]) in keyset and k != ks[i])
        synf.append((k, city[i], fq[i]) in keyform and k != ks[i])
    res.append((np.mean(syn), np.mean(synf)))
syn = np.mean([r[0] for r in res]); synf = np.mean([r[1] for r in res])
acc = f["acc"].to_numpy()
print(f"swap|nsame n={len(kq)}: observed twin rate {obs.mean():.4f} (same forms {obsf.mean():.4f}); synthetic random-descriptor twin rate {syn:.4f} (same forms {synf:.4f})")
print(f"   accepted: obs {obs[acc].mean():.4f} / unaccepted: obs {obs[~acc].mean():.4f}")
# per p3 bucket
p3 = f["p3"].to_numpy(); g = f["p2g"].to_numpy()
for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, .99), (.99, 1.01)]:
    m = (p3 >= lo) & (p3 < hi)
    print(f"   p3 in [{lo},{hi}): n={m.sum()} twin {obs[m].mean():.4f} (forms {obsf[m].mean():.4f})")
