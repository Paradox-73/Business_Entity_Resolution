"""Lowercase fingerprint on the France pairs each uploaded version added/removed vs v7ens; compare with LB-implied true rates."""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
NC = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"
OUT = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/refute_desc"
a = pl.read_parquet(f"{OUT}/fr_low.parquet")
Ld, Lt = 0.0338, 0.0008
E = pl.read_parquet(f"{FR}/pairs_v7ens.parquet").select("q", "s")
M = pl.read_parquet(f"{NC}/pairs_v7m.parquet").select("q", "s")
vers = {"v7m": M}
for nm in ("v7i", "v7j", "v7g_num"):
    vers[nm] = pl.read_parquet(f"{FR}/pairs_{nm}.parquet").select("q", "s")
for nm, V in vers.items():
    for lab, x in (("ADDED", V.join(E, on=["q", "s"], how="anti")), ("REMOVED", E.join(V, on=["q", "s"], how="anti"))):
        y = x.join(a, on=["q", "s"], how="left")
        tot = x.height; notbest = y["low"].null_count()
        w = y.filter(pl.col("kind").is_in(["swap", "added"]))
        L = w["low"].mean() if w.height else float("nan")
        t = (Ld - L) / (Ld - Lt) if w.height else float("nan")
        se = np.sqrt(max(L, 1e-4) * (1 - L) / max(w.height, 1)) / (Ld - Lt)
        print(f"{nm:8s} {lab:7s} pairs {tot:6d} (not best cand {notbest}); swap/added {w.height:6d}: lower {L:.4f} -> true share {t:.2f}+-{se:.2f}")
        g = (y.filter(pl.col("kind").is_in(["swap", "added"])).group_by("cls", "num").agg(n=pl.len(), low=pl.col("low").mean())
             .with_columns(t=((Ld - pl.col("low")) / (Ld - Lt)).round(2)).filter(pl.col("n") >= 300).sort("n", descending=True))
        print("    ", g.to_dicts()[:8])
        print("     other kinds:", y.filter(~pl.col("kind").is_in(["swap", "added"])).group_by("kind", "num").len().sort("len", descending=True).head(6).to_dicts())
