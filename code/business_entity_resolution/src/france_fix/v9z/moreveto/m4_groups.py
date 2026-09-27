"""Candidate false-accept groups among currently accepted France pairs; lowercase test on the case-blind model's accepts."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(50)
OUT = f"{SCRATCH}/frfix3/moreveto"
NC = f"{SCRATCH}/frfix/namechg"
FR = f"{SCRATCH}/france"
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "qn", "sn", "pat", "acc", "p2", "p2g"])
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q", "s", "ops", "nc", "num"])
c = pl.read_parquet(f"{NC}/fr_chg.parquet", columns=["q", "s", "p3", "added", "dropped"])
t = t.join(o, on=["q", "s"]).join(c, on=["q", "s"], how="left")
cur = pl.read_parquet(f"{OUT}/acc_v9y.parquet", columns=["q", "s"]).with_columns(cur=pl.lit(True))
dv = pl.read_parquet(f"{NC}/veto_set.parquet", columns=["q", "s"]).with_columns(dv=pl.lit(True))
ad = pl.read_parquet(f"{SCRATCH}/frfix/build/add_set_v2_final.parquet", columns=["q", "s"]).with_columns(ad=pl.lit(True))
fp = pl.read_parquet(f"{WORK}/frfix2/fp_veto_set.parquet", columns=["q", "s"]).with_columns(fv=pl.lit(True))
t = (t.join(cur, on=["q", "s"], how="left").join(dv, on=["q", "s"], how="left").join(ad, on=["q", "s"], how="left")
      .join(fp, on=["q", "s"], how="left").with_columns(pl.col("cur", "dv", "ad", "fv").fill_null(False)))
t = t.with_columns(pop=(pl.col("acc") & ~pl.col("dv")) | pl.col("ad"),
                   handle=~pl.col("qn").fill_null("").str.contains(" ") & pl.col("sn").fill_null("").str.contains(" "),
                   kind=pl.col("pat").str.split("|").list.first(), npat=pl.col("pat").str.split("|").list.last())
t = t.with_columns(lowx=pl.col("nc").list.contains("LOWER") & ~pl.col("handle"),
                   sig=pl.col("nc").list.filter(pl.element() != "LOWER").list.join("+"))
w = pl.read_parquet(f"{NC}/word_class.parquet")
wc = dict(zip(w["a"].to_list(), w["cls"].to_list())); wk = dict(zip(w["a"].to_list(), w["keep"].to_list()))
wn = dict(zip(w["a"].to_list(), w["n"].to_list())); wd = dict(zip(w["a"].to_list(), w["df_s1"].to_list()))
def lab(r):
    a, d, k = r["added"] or [], r["dropped"] or [], r["kind"]
    if r["dv"]:
        return "vetoed_D"
    if k in ("same", "squashed", "acronym", "empty") or r["handle"]:
        return "nochg/handle"
    ca = [wc.get(x, "rare") for x in a]; cd = [wc.get(x, "rare") for x in d]
    if k == "other":
        return "other_name"
    if not a:
        return "drop_D" if "D" in cd else "drop_only"
    if len(a) >= 2:
        return "add2+" + ("_D" if "D" in ca else "")
    x = a[0]; cl = ca[0]; kp = wk.get(x); n = wn.get(x, 0); df = wd.get(x, 0) or 0
    pre = "swap" if d else "add"
    if cl == "D":
        return pre + "_D_notveto"
    if cl in ("X", "G", "N"):
        return pre + "_" + cl + ("_dropD" if "D" in cd else "")
    if cl == "M":
        return pre + "_M" + ("lo" if kp < 0.25 else "mid" if kp < 0.5 else "hi")
    # rare word
    if n >= 8 and kp is not None and 0.15 <= kp < 0.6 and df >= 30:
        return pre + "_rare_desclike"
    if df >= 100:
        return pre + "_rare_commonS1"
    return pre + "_rare_other" + ("_dropD" if "D" in cd else "")
ch = t.filter(pl.col("pop") | pl.col("cur") | pl.col("dv"))
ch = ch.with_columns(grp=pl.Series([lab(r) for r in ch.select("added", "dropped", "kind", "dv", "handle").iter_rows(named=True)]))
ch.drop("ops").write_parquet(f"{OUT}/grp_table.parquet")
TL, DL = 0.002, 0.0338
g = (ch.filter(pl.col("pop")).group_by("grp", "npat").agg(n_pop=pl.len(), n_cur=pl.col("cur").sum(), nlow=pl.col("lowx").sum(),
        low=pl.col("lowx").mean(), p3=pl.col("p3").mean(), g=pl.col("p2g").mean())
     .with_columns(t_est=((DL - pl.col("low")) / (DL - TL))).sort("n_cur", descending=True))
print(g.filter(pl.col("n_pop") >= 30))
# the vetoed set itself for reference (population = v7ens acc)
print("reference: descriptor veto set lowx share", ch.filter(pl.col("dv") & pl.col("acc"))["lowx"].mean(), ch.filter(pl.col("dv") & pl.col("acc")).height)
