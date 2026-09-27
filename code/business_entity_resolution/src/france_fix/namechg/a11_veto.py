"""Define the veto set: accepted France best-candidate pairs whose record replaces/adds a descriptor-class (D) word."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
OUT = f"{SCRATCH}/frfix/namechg"; FR = f"{SCRATCH}/france"
a = pl.read_parquet(f"{OUT}/fr_all_cls.parquet")
oth = pl.read_parquet(f"{OUT}/fr_oth.parquet")   # descriptor-S1 restricted stat is in a8; recompute here on all S1
a = a.with_columns(sameacc=((pl.col("cls") == "samename") & pl.col("acc")))
a = a.with_columns(oth=pl.col("sameacc").sum().over("s") - pl.col("sameacc").cast(pl.Int64),
                   kind=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last())
T = a.filter((pl.col("cls") == "samename") & (pl.col("num") == "nsame") & pl.col("acc"))
Dc = a.filter((pl.col("cls") == "X") & (pl.col("num") == "ndiff") | ((pl.col("cls") == "G") & (pl.col("kind") == "added") & (pl.col("num") == "ndiff")))
def st(x):
    o = x["oth"].to_numpy().astype(float); return o.mean(), o.std()/np.sqrt(len(o)), (o == 0).mean(), np.sqrt((o == 0).mean()*(1-(o == 0).mean())/len(o))
tm, _, t0, _ = st(T); dm, _, d0, _ = st(Dc)
def est(x, name):
    m, s, z, zs = st(x)
    tmn, tz = (dm-m)/(dm-tm), (d0-z)/(d0-t0); sm, sz = s/(dm-tm), zs/(d0-t0)
    w = np.array([1/sm**2, 1/sz**2]); comb = (w[0]*tmn+w[1]*tz)/w.sum()
    print(f"{name:52s} n={x.height:6d} true share: mean {tmn:.2f}+-{sm:.2f} zero {tz:.2f}+-{sz:.2f} combined {comb:.2f}+-{1/np.sqrt(w.sum()):.2f}")
c = pl.read_parquet(f"{OUT}/fr_chg.parquet", columns=["q", "added", "dropped", "qn", "sn", "twin_samenum", "s_2", "p2_2"])
d = a.filter(pl.col("cls") == "D").join(c, on="q")
# synonym / abbreviation maps: (dropped, added) single-word swaps whose number keeping is noise-like (>= 0.6, n >= 30) or lift >= 5
sm_ = pl.read_parquet(f"{OUT}/swap_matrix.parquet")
syn = sm_.filter(((pl.col("keep") >= 0.6) & (pl.col("nk") >= 30)) | (pl.col("lift") >= 5)).select("d", "a")
print("D-class (dropped,added) pairs treated as synonyms:", syn.join(d.filter(pl.col("added").list.len() == 1).select(
    d=pl.col("dropped").list.first(), a=pl.col("added").list.first()).unique(), on=["d", "a"]).to_dicts()[:40])
d = d.with_columns(d1=pl.col("dropped").list.first(), a1=pl.col("added").list.first())
d = d.join(syn.with_columns(syn=pl.lit(True)).rename({"d": "d1", "a": "a1"}), on=["d1", "a1"], how="left").with_columns(pl.col("syn").fill_null(False))
# stem variants (sport/sportive, culturel/culturelle ...): an added word that shares a 5-letter prefix with a dropped word
d = d.with_columns(stem=pl.struct("added", "dropped").map_elements(
    lambda r: any(x[:5] == y[:5] for x in r["added"] for y in r["dropped"] if len(x) >= 5 and len(y) >= 5), return_dtype=pl.Boolean))
print("D rows", d.height, "synonym rows", d["syn"].sum(), "stem rows", d["stem"].sum())
print(d.group_by("num", "acc").agg(n=pl.len(), syn=pl.col("syn").sum(), stem=pl.col("stem").sum()).sort("num", "acc"))
est(d.filter(pl.col("syn") | pl.col("stem")), "D synonym/stem rows (all statuses)")
est(d.filter(~pl.col("syn") & ~pl.col("stem") & (pl.col("num") == "nsame")), "D non-synonym nsame (all statuses)")
for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)):
    est(d.filter(~pl.col("syn") & ~pl.col("stem") & (pl.col("num") == "nsame") & (pl.col("p3") >= lo) & (pl.col("p3") < hi)), f"  D nsame all statuses, p3 [{lo},{hi})")
for k in ("swap", "other", "added"):
    x = d.filter(~pl.col("syn") & ~pl.col("stem") & (pl.col("num") == "nsame") & (pl.col("kind") == k))
    if x.height > 1000: est(x, f"  D nsame all statuses kind={k}")
V = d.filter(pl.col("acc") & ~pl.col("syn") & ~pl.col("stem"))
print("VETO SET (accepted D, not synonym/stem):", V.height, V.group_by("num", "kind").len().sort("len", descending=True).to_dicts())
print("   S1 rows touched", V["s"].n_unique(), "; twin S1 with record's name at same number:", V["twin_samenum"].sum(),
      "; record has a 2nd candidate with p2 >= 0.5:", (V["p2_2"] >= 0.5).sum())
print("   p3 bands", V.group_by(pl.col("p3").cut([0.5, 0.9, 0.99, 0.999])).len().sort("p3").to_dicts())
print("   top words", V.group_by("a1").len().sort("len", descending=True).head(25).to_dicts())
V.select("q", "s", "num", "kind", "p2", "p3", "p2g", "qn", "sn").write_parquet(f"{OUT}/veto_set.parquet")
print(V.sample(20, seed=3).select("sn", "qn", "p3", "p2g", "num"))
