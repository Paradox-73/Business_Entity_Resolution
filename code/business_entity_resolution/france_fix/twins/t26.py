# twin-excess rate per subset -> label-free false-share estimate, anchored on v7m restore set (LB: ~40% true)
import polars as pl, numpy as np
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
top = pl.read_parquet(f"{T}/top_k3.parquet")
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
keyc = set(zip(s1["ks"].to_list(), s1["scity"].to_list()))
def excess(g, reps=6, seed=5):
    kq = g["kq"].to_list(); ks = g["ks"].to_list(); city = g["qcity"].to_list()
    shared, new = [], []
    for a, b in zip(kq, ks):
        A, B = set(a.split()), set(b.split()); shared.append(A & B); new.append(sorted(A - B))
    oc = sum((k, c) in keyc for k, c in zip(kq, city))
    rng = np.random.default_rng(seed); sc = []
    for rep in range(reps):
        perm = rng.permutation(len(new))
        sc.append(sum((" ".join(sorted(shared[i] | set(new[j]))), city[i]) in keyc and " ".join(sorted(shared[i] | set(new[j]))) != ks[i] for i, j in enumerate(perm)))
    n = len(kq); return n, oc / n, np.mean(sc) / n, np.std(sc) / n
f = top.filter(pl.col("nk").is_in(["swap"]))
f = f.with_columns(rest=(pl.col("pr") >= 0.5) & ~pl.col("acc"))
rows = []
subsets = {
  "v7m restore set (swap|nsame)": f.filter(pl.col("rest") & (pl.col("num") == "nsame")),
  "unacc swap|nsame not restored": f.filter(~pl.col("acc") & ~pl.col("rest") & (pl.col("num") == "nsame")),
}
a = f.filter(pl.col("acc") & (pl.col("num") == "nsame"))
for lo, hi in [(0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 0.99), (0.99, 0.999), (0.999, 1.01)]:
    subsets[f"acc swap|nsame pf[{lo},{hi})"] = a.filter((pl.col("p2") >= lo) & (pl.col("p2") < hi))
for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]:
    subsets[f"acc swap|nsame p3[{lo},{hi})"] = a.filter((pl.col("p3") >= lo) & (pl.col("p3") < hi))
subsets["acc swap|nmiss"] = f.filter(pl.col("acc") & (pl.col("num") == "nmiss"))
for k, g in [(k, g) for k, g in subsets.items() if g.height > 0]:
    n, o, s, sd = excess(g)
    rows.append((k, n, o, s, o - s, sd))
    print(f"{k:38s} n={n:6d} twin {o:.4f} null {s:.4f} excess {o - s:+.4f} (+-{sd:.4f})")
anchor = [r for r in rows if r[0].startswith("v7m")][0]
D = anchor[4] / 0.60
print(f"anchor: v7m restore set false share 0.60 (LB) -> excess per unit false share D={D:.4f}")
for r in rows:
    print(f"   {r[0]:38s} est false share {r[4] / D:+.3f} -> est true rate {1 - r[4] / D:.3f}  (n={r[1]})")
