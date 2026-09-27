import polars as pl, numpy as np, random
from collections import defaultdict
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
top = pl.read_parquet(f"{T}/top_k3.parquet", columns=["q", "s", "p2", "p2g", "p3", "pr", "acc", "pat", "kq", "ks", "qcity", "qcity_right"])
print("qcity == qcity_right:", (top["qcity"] == top["qcity_right"]).mean())
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyc = set(zip(s1["ks"].to_list(), s1["scity"].to_list()))
f = top.filter(pl.col("pat") == "swap|nsame")
print("swap|nsame records:", f.height, " accepted:", f["acc"].sum())
kq = f["kq"].to_list(); ks = f["ks"].to_list(); city = f["qcity"].to_list()
obs = np.array([(a, c) in keyc and a != b for a, b, c in zip(kq, ks, city)])
print("observed twin:", obs.sum(), obs.mean())
# independent null: new-word set replaced by a new-word set drawn from records in the SAME city with the SAME number of new words, within subset
shared = []; new = []
for a, b in zip(kq, ks):
    A, B = set(a.split()), set(b.split()); shared.append(A & B); new.append(frozenset(A - B))
acc = f["acc"].to_numpy(); p3 = f["p3"].to_numpy(); pf = f["p2"].to_numpy(); pr = f["pr"].to_numpy()
def null_count(idx, reps=5, seed=11):
    rng = random.Random(seed)
    pool = defaultdict(list)
    for i in idx: pool[(city[i], len(new[i]))].append(i)
    out = []
    for r in range(reps):
        c = 0
        for i in idx:
            j = rng.choice(pool[(city[i], len(new[i]))])
            k = " ".join(sorted(shared[i] | new[j]))
            c += (k, city[i]) in keyc and k != ks[i]
        out.append(c)
    return np.mean(out), np.std(out)
allidx = np.arange(len(kq))
rest = (pr >= 0.5) & ~acc
subs = {"all": allidx, "accepted": allidx[acc], "unaccepted": allidx[~acc], "v7m restore set": allidx[rest],
        "unacc not restored": allidx[~acc & ~rest]}
for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, .99), (.99, 1.01)]:
    subs[f"acc p3[{lo},{hi})"] = allidx[acc & (p3 >= lo) & (p3 < hi)]
for lo, hi in [(0.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
    subs[f"acc pf[{lo},{hi})"] = allidx[acc & (pf >= lo) & (pf < hi)]
for k, idx in subs.items():
    if len(idx) == 0: continue
    m, sd = null_count(idx)
    o = obs[idx].sum()
    print(f"{k:22s} n={len(idx):7d} obs {o:6d} ({o/len(idx):.4f}) null(same city,same #new) {m:8.1f} +-{sd:5.1f} ({m/len(idx):.4f}) excess {o-m:+8.1f} = {(o-m)/len(idx):+.4f}")
