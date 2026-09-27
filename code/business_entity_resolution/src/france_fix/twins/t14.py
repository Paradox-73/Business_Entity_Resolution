import polars as pl, sys, re, unicodedata
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from france_cal import toks, FR_FORMS
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(42); pl.Config.set_tbl_width_chars(330)
FS = set(FR_FORMS)
def forms(x):
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\b([a-z])\.", r"\1", x)
    w = re.sub(r"[^a-z0-9]+", " ", x).split()
    w = ["sarl" if t == "5arl" else "sas" if t == "5as" else t for t in w]
    return " ".join(sorted(set(t for t in w if t in FS)))
top = pl.read_parquet(f"{T}/top_k2.parquet").drop("nx_all", "nx_city", "xin")
top = top.with_columns(fq=pl.Series([forms(x) for x in top["qn"].to_list()]))
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet")
s1 = s1.with_columns(ks=pl.Series([" ".join(sorted(set(toks(x)))) for x in s1["sn"].to_list()]), fs=pl.Series([forms(x) for x in s1["sn"].to_list()]))
s1.select("s", "ks", "fs", "scity", "snum", "sst").write_parquet(f"{T}/fr_s1kk.parquet")
rc = pl.read_parquet(f"{T}/fr_recsk.parquet", columns=["q", "qcity", "qnum", "qst"])
top = top.join(rc, on="q", how="left")
# twin candidates: same toks-name in same city, different S1 than best; legal forms compatible (record has no form, or S1 has no form, or share one)
tw = top.select("q", "s", "kq", "fq", "qcity").join(s1.select(sx="s", ks="ks", fs="fs", scity="scity"), left_on=["kq", "qcity"], right_on=["ks", "scity"])
tw = tw.filter(pl.col("sx") != pl.col("s"))
tw = tw.with_columns(compat=(pl.col("fq") == "") | (pl.col("fs") == "") | (pl.col("fq").str.split(" ").list.set_intersection(pl.col("fs").str.split(" ")).list.len() > 0),
                     exactf=pl.col("fq") == pl.col("fs"))
agg = tw.group_by("q").agg(n_tw=pl.len(), n_compat=pl.col("compat").sum(), n_exactf=pl.col("exactf").sum())
top = top.join(agg, on="q", how="left").with_columns(pl.col("n_tw").fill_null(0), pl.col("n_compat").fill_null(0), pl.col("n_exactf").fill_null(0))
top.write_parquet(f"{T}/top_k3.parquet")
tw.write_parquet(f"{T}/twins_all.parquet")
f = top.filter(pl.col("pat") == "swap|nsame")
print(f.group_by(pl.col("n_compat").clip(0, 3).alias("ncompat"), pl.col("n_exactf").clip(0, 1).alias("exf")).agg(n=pl.len(), acc=pl.col("acc").mean(), nacc=pl.col("acc").sum(), p2=pl.col("p2").mean(), g=pl.col("p2g").mean(), p3=pl.col("p3").mean()).sort("ncompat", "exf"))
