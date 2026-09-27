"""Veto: descriptor-class (D) swaps that the first veto kept as 'synonym/stem' but are not abbreviations
(sport<->sportive, college<->collectif, trade word -> D word ...). Evidence, gain estimate, scores file, decision check."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import numpy as np
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/fp"))
from pipeline import decide_expf
from fp1_dec import france_scores
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240); pl.Config.set_fmt_str_lengths(45)
OUT = f"{SCRATCH}/frfix3/moreveto"
NC = f"{SCRATCH}/frfix/namechg"
CUR = f"{WORK}/frfix2/test_scores_v9b_fpveto.parquet"
NEW = f"{WORK}/frfix3/test_scores_v9y_moreveto.parquet"
g = pl.read_parquet(f"{OUT}/grp_table2.parquet")
w = pl.read_parquet(f"{NC}/word_class.parquet")
wk = dict(zip(w["a"].to_list(), w["keep"].to_list()))
KEEP_ADDED = {"compagnie", "etablissements", "etablissement"}      # expansions of cie / ets (873 + 16 rows, 0 lowercase)


def abbrev(added, dropped):
    for a in added or []:
        if a in KEEP_ADDED:
            return True
        for d in dropped or []:
            sh, lo = (a, d) if len(a) <= len(d) else (d, a)
            if lo.startswith(sh) and len(sh) <= 4:        # auto/automobile, tech/technologies
                return True
    return False


x = g.filter(pl.col("grp").is_in(["swap_D_notveto", "add2+_D"]) & (pl.col("pop") | pl.col("cur")))
x = x.with_columns(abbr=pl.Series([abbrev(a, d) for a, d in zip(x["added"].to_list(), x["dropped"].to_list())], dtype=pl.Boolean))
x = x.with_columns(stem=pl.struct("added", "dropped").map_elements(
    lambda r: any(p[:5] == q[:5] for p in r["added"] for q in r["dropped"] if len(p) >= 5 and len(q) >= 5), return_dtype=pl.Boolean),
    keep=pl.col("added").list.first().replace_strict(wk, default=None))
TL, DL = 0.0005, 0.0338
print("lowercase test (population = case-blind model accepts, not vetoed; lowercase = all-lowercase record, handle forms excluded)")
r = (x.filter(pl.col("pop")).group_by("grp", "abbr", "stem").agg(n_pop=pl.len(), n_cur=pl.col("cur").sum(), nlow=pl.col("lowx").sum(),
        keep=pl.col("keep").mean()).with_columns(low=pl.col("nlow") / pl.col("n_pop")).with_columns(t_est=(DL - pl.col("low")) / (DL - TL)).sort("n_pop", descending=True))
print(r)
V = x.filter(pl.col("cur") & ~pl.col("abbr"))
P = x.filter(pl.col("pop") & ~pl.col("abbr"))
nl, n = int(P["lowx"].sum()), P.height
lo, hi = [float(v) for v in (np.array([max(nl - 1.96 * np.sqrt(nl), 0), nl + 1.96 * np.sqrt(nl) + 1.9]) / n)]
print(f"VETO SET: {V.height} accepted pairs (population {n}, lowercase {nl} = {nl / n:.4f}; references true {TL}, fake {DL}); "
      f"true share {(DL - nl / n) / (DL - TL):.2f} (approx 95% range {(DL - hi) / (DL - TL):.2f}..{min((DL - lo) / (DL - TL), 1):.2f})")
print("  by npat", V.group_by("npat").len().to_dicts())
print("  top (dropped -> added)", V.group_by(pl.col("dropped").list.join(" "), pl.col("added").list.join(" ")).len().sort("len", descending=True).head(20).to_dicts())
print(V.select("sn", "qn", "npat", "p3").sample(min(15, V.height), seed=4))
t = max((DL - nl / n) / (DL - TL), 0.0)
# expected France macro gain: each vetoed pair on an S1 with k accepted records
acc = pl.read_parquet(f"{OUT}/acc_v9y.parquet", columns=["s", "q"])
k = acc.group_by("s").len("k")
VV = V.join(k, on="s", how="left")
def f05(p, r):
    return 0.0 if p == 0 or r == 0 else 1.25 * p * r / (0.25 * p + r)
def pair(kk, t, pe=0.75):
    if kk == 1:
        return (1 - t) * pe - t * 1.0
    return (1 - t) * (1 - f05((kk - 1) / kk, 1.0)) - t * (1 - f05(1.0, (kk - 1) / kk))
nS1 = 259452
for tt in (t, 0.3, 0.5):
    gsum = sum(pair(kk, tt) for kk in VV["k"].to_list())
    print(f"France macro gain at true share {tt:.2f}: {gsum / nS1:+.6f}  (LB x0.15: {0.15 * gsum / nS1:+.7f})")
V.select("q", "s", "sn", "qn", "npat", "p2", "p3", "lowx").write_parquet(f"{WORK}/frfix3/moreveto_set.parquet")
# scores file
b = pl.read_parquet(CUR)
vv = V.select("q", "s").with_columns(_v=pl.lit(True))
b = b.join(vv, on=["q", "s"], how="left").with_columns(
    p2=pl.when(pl.col("_v").fill_null(False)).then(0.0).otherwise(pl.col("p2")).cast(pl.Float32)).drop("_v")
b.write_parquet(NEW)
print("wrote", NEW, b.height, "rows; p2 set to 0 on", V.height)
del b
nb = france_scores(NEW)
m = decide_expf(nb, "p2", 0.5, 1.0)
old = acc
rem = old.join(m, on=["s", "q"], how="anti"); new = m.join(old, on=["s", "q"], how="anti")
print("France decision:", m.height, "pairs (v9y", old.height, "); removed", rem.height, "(in veto", rem.join(V.select("s", "q"), on=["s", "q"]).height,
      "); added", new.height)
print("added pairs (a vetoed record moving to its 2nd candidate?):", new.head(10).to_dicts())
