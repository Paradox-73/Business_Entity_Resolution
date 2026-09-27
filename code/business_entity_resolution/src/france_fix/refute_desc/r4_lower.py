"""Lowercase fingerprint: share of records whose raw name is entirely lowercase. US/India labels: true word-change records
0.25-0.37%, decoys 2.0-4.1%. France: estimate true share of name-change groups as (Ld - L)/(Ld - Lt)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
FR = f"{SCRATCH}/france"
NC = f"{SCRATCH}/frfix/namechg"
OUT = f"{SCRATCH}/frfix/refute_desc"
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "qn"])
a = pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "p3", "p2g", "acc", "in7m", "cls", "pat", "num"])
a = a.join(t, on=["q", "s"]).with_columns(low=pl.col("qn") == pl.col("qn").str.to_lowercase(),
                                           hasalpha=pl.col("qn").str.contains(r"[A-Za-z]"),
                                           kind=pl.col("pat").str.split("|").list.first())
a = a.filter(pl.col("hasalpha"))
v = pl.read_parquet(f"{NC}/veto_set.parquet", columns=["q", "s"]).with_columns(veto=pl.lit(True))
a = a.join(v, on=["q", "s"], how="left").with_columns(pl.col("veto").fill_null(False))
a.select("q", "s", "low", "cls", "kind", "num", "acc", "in7m", "veto", "p3", "p2g").write_parquet(f"{OUT}/fr_low.parquet")
chg = a.filter(pl.col("kind").is_in(["swap", "added", "other", "dropped"]))
Ld = chg.filter(pl.col("cls").is_in(["X"]))["low"].mean()
Lt = chg.filter(pl.col("cls").is_in(["N"]) & (pl.col("num") == "nsame"))["low"].mean()
print(f"refs: decoy X all numbers {Ld:.4f} (n {chg.filter(pl.col('cls')=='X').height}); true N nsame {Lt:.4f}")


def est(x, name):
    n = x.height
    if n == 0:
        return
    L = x["low"].mean(); se = np.sqrt(max(L, 1e-4) * (1 - L) / n)
    print(f"{name:55s} n={n:7d} lower {L:.4f}+-{se:.4f} -> true share {(Ld - L) / (Ld - Lt):5.2f} +-{se / (Ld - Lt):.2f}")


print("--- by class / kind / number (all statuses)")
for cls in ["X", "N", "G", "D", "rare", "M", "drop"]:
    for num in ["nsame", "ndiff", "nmiss"]:
        est(chg.filter((pl.col("cls") == cls) & (pl.col("num") == num)), f"{cls} {num}")
print("--- D nsame by status")
D = chg.filter((pl.col("cls") == "D") & (pl.col("num") == "nsame"))
est(D.filter(pl.col("veto")), "VETO SET (accepted D)")
est(D.filter(pl.col("in7m") & ~pl.col("acc")), "D nsame restored in v7m")
est(D.filter(~pl.col("in7m") & ~pl.col("acc")), "D nsame rejected in v7m too")
V = chg.filter(pl.col("veto"))
for lo, hi in ((0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)):
    est(V.filter((pl.col("p3") >= lo) & (pl.col("p3") < hi)), f"  veto p3 [{lo},{hi})")
for lo, hi in ((0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
    est(V.filter((pl.col("p2g") >= lo) & (pl.col("p2g") < hi)), f"  veto g [{lo},{hi})")
est(a.filter(pl.col("veto")), "VETO SET all kinds")
print("--- G nsame by status")
G = chg.filter((pl.col("cls") == "G") & (pl.col("num") == "nsame"))
est(G.filter(pl.col("acc")), "G nsame accepted"); est(G.filter(pl.col("in7m") & ~pl.col("acc")), "G nsame restored"); est(G.filter(~pl.col("in7m") & ~pl.col("acc")), "G nsame rejected")
N = chg.filter((pl.col("cls") == "N") & (pl.col("num") == "nsame"))
est(N.filter(pl.col("acc")), "N nsame accepted"); est(N.filter(~pl.col("acc")), "N nsame not accepted")
print("--- same-name records (different reference, for scale)")
S = a.filter(pl.col("kind") == "same")
for num in ["nsame", "ndiff"]:
    for acc in [True, False]:
        x = S.filter((pl.col("num") == num) & (pl.col("acc") == acc)); print(f"same {num} acc={acc} n={x.height} lower {x['low'].mean():.4f}")
