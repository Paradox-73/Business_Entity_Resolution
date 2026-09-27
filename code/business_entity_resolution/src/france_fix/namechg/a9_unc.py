import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
exec(open("a8_mix.py").read().split('for cls in ("D"')[0])
print("--- unconditional (all statuses), same number")
for cls in ("samename", "drop", "N", "G", "D", "rare", "X", "M"):
    for kind in ("swap", "added", "same", "dropped", "other", None):
        x = a.filter((pl.col("cls") == cls) & (pl.col("num") == "nsame"))
        if kind: x = x.filter(pl.col("kind") == kind)
        if x.height >= 1500: est(x, f"{cls} {kind or 'ALL'} nsame all")
print("--- D nsame all by word (top)")
c = pl.read_parquet(f"{OUT}/fr_chg.parquet", columns=["q", "added"])
x = a.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame")).join(c, on="q").with_columns(w=pl.col("added").list.first())
for w in ["club", "amicale", "ecole", "comite", "centre", "amis", "sportive", "parents", "union", "societe", "federation"]:
    est(x.filter(pl.col("w") == w), f"D nsame all word={w}")
# model-accepted-looking words vs rejected-looking
acc_rate = x.group_by("w").agg(ar=pl.col("acc").mean(), n=pl.len()).filter(pl.col("n") >= 300)
hi = acc_rate.filter(pl.col("ar") >= 0.6)["w"].to_list(); lo = acc_rate.filter(pl.col("ar") < 0.3)["w"].to_list()
print("high-acceptance words", hi); print("low-acceptance words", lo)
est(x.filter(pl.col("w").is_in(hi)), "D nsame all, words the model accepts >=60%")
est(x.filter(pl.col("w").is_in(lo)), "D nsame all, words the model accepts <30%")
