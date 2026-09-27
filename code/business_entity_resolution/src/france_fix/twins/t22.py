# duplicate records (same normalized name+forms+house number+street+city): do they share fate?
import polars as pl, sys, re, unicodedata
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from france_cal import toks
from fr_restore import street
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(330)
top = pl.read_parquet(f"{T}/top_k3.parquet", columns=["q", "s", "p2", "p2g", "p3", "acc", "pat", "kq", "fq", "qnum", "qst", "qcity", "qn", "qa", "sn", "sa"])
top = top.with_columns(dk=pl.concat_str([pl.col("kq"), pl.col("fq"), pl.col("qnum").fill_null("-"), pl.col("qst"), pl.col("qcity")], separator="|"))
top = top.with_columns(gsz=pl.len().over("dk"), nacc=pl.col("acc").sum().over("dk"), nS=pl.col("s").n_unique().over("dk"),
                       nSacc=pl.col("s").filter(pl.col("acc")).n_unique().over("dk"))
d = top.filter(pl.col("gsz") >= 2)
print("France records in duplicate groups:", d.height, " groups:", d["dk"].n_unique())
g = d.group_by("dk").agg(gsz=pl.len(), nacc=pl.col("acc").sum(), nS=pl.col("s").n_unique())
g = g.with_columns(state=pl.when(pl.col("nacc") == 0).then(pl.lit("none")).when(pl.col("nacc") == pl.col("gsz")).then(pl.when(pl.col("nS") == 1).then(pl.lit("all_same_S1")).otherwise(pl.lit("all_acc_diff_S1"))).otherwise(pl.lit("mixed")))
print(g.group_by("state").agg(groups=pl.len(), recs=pl.col("gsz").sum(), acc=pl.col("nacc").sum()))
mixed = d.filter((pl.col("nacc") > 0) & (pl.col("nacc") < pl.col("gsz")))
print("mixed groups: unaccepted members:", mixed.filter(~pl.col("acc")).height, " of which best S1 == the accepted S1 of the group:",
      mixed.filter(~pl.col("acc") & (pl.col("nSacc") == 1)).join(mixed.filter(pl.col("acc")).select("dk", sacc="s").unique(), on="dk").filter(pl.col("s") == pl.col("sacc")).height)
print(mixed.filter(~pl.col("acc")).group_by("pat").agg(n=pl.len(), p2=pl.col("p2").mean(), g=pl.col("p2g").mean(), p3=pl.col("p3").mean()).sort("n", descending=True).head(10))
print(mixed.sort("dk").head(24).select("qn", "qa", "sn", "p2", "p2g", "p3", "acc", "pat"))
d.write_parquet(f"{T}/dups.parquet")
