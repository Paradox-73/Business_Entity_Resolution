"""Address-equality fingerprint (case/space-insensitive equality of record and S1 address), restricted to same number AND
same street, France groups; plus street-level detail. Label check on US/India in r4."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
from rapidfuzz import fuzz
from fr_restore import street
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
FR = f"{SCRATCH}/france"
NC = f"{SCRATCH}/frfix/namechg"
OUT = f"{SCRATCH}/frfix/refute_desc"
f = pl.read_parquet(f"{OUT}/fr_feat.parquet", columns=["q", "s", "grp", "a_ci", "a_exact"])
f = f.filter(~pl.col("grp").is_in(["D_ndiff", "G_ndiff", "X_ndiff", "same_ndiff"]))
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "qa", "sa", "pat"]).join(f.select("q"), on="q")
t = t.filter(pl.col("pat").str.ends_with("nsame"))
ss = [street(a) for a in t["sa"].to_list()]; qs = [street(a) for a in t["qa"].to_list()]
sm = [bool(a[0] is not None and a[0] == c[0] and a[1] and c[1] and fuzz.token_sort_ratio(a[1], c[1]) >= 85) for a, c in zip(ss, qs)]
t = t.with_columns(samestreet=pl.Series(sm))
f = f.join(t.select("q", "samestreet"), on="q")
a = pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "acc", "p3", "p2g"])
f = f.join(a, on="q")
f.write_parquet(f"{OUT}/fr_addr.parquet")
r = f.group_by("grp", "samestreet").agg(n=pl.len(), a_ci=pl.col("a_ci").mean(), a_exact=pl.col("a_exact").mean(), acc=pl.col("acc").mean()).sort("grp", "samestreet")
print(r)
# all-status D nsame
d = f.filter(pl.col("grp").str.starts_with("D_nsame") | (pl.col("grp") == "D_veto(acc)"))
print("D nsame ALL statuses:", d.group_by("samestreet").agg(n=pl.len(), a_ci=pl.col("a_ci").mean(), acc=pl.col("acc").mean()))
