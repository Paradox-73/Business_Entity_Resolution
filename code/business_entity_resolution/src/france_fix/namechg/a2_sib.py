"""Sibling test: records sharing the same changed name for the same best S1. US/India labelled vs France."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
import france_cal as fc
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
FR = f"{SCRATCH}/france"
OUT = f"{SCRATCH}/frfix/namechg"


def key(x):
    return " ".join(sorted(set(fc.toks(x))))


u = pl.read_parquet(f"{FR}/usi_top.parquet", columns=["q", "s", "label", "p", "p3", "p2g", "qn", "sn", "pat", "country"])
u = u.with_columns(qk=pl.Series([key(x) for x in u["qn"].to_list()]), sk=pl.Series([key(x) for x in u["sn"].to_list()]))
u = u.with_columns(nsib=pl.len().over("s", "qk"), nrec=pl.len().over("s"),
                   nsame=(pl.col("qk") == pl.col("sk")).sum().over("s"))
u = u.with_columns(chg=pl.col("qk") != pl.col("sk"))
x = u.filter(pl.col("pat").is_in(["swap|nsame", "added|nsame", "other|nsame", "dropped|nsame"]))
print("US/India name change same number: truth by number of records sharing the changed name (same best S1)")
print(x.group_by("country", "pat", pl.col("nsib").clip(1, 4)).agg(n=pl.len(), true=pl.col("label").mean(), acc=(pl.col("p") >= 0.5).mean())
      .sort("country", "pat", "nsib"))
u.select("q", "s", "qk", "sk", "nsib", "nrec", "nsame").write_parquet(f"{OUT}/usi_keys.parquet")

f = pl.read_parquet(f"{OUT}/fr_all.parquet")
c = pl.read_parquet(f"{OUT}/fr_chg.parquet", columns=["q", "qk", "sn"])
# France keys for all best candidates (same-name rows have qk == sk; compute for all rows)
top = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "qn", "sn"])
top = top.with_columns(qk=pl.Series([key(x) for x in top["qn"].to_list()]), sk=pl.Series([key(x) for x in top["sn"].to_list()]))
f = f.join(top.select("q", "qk", "sk"), on="q")
f = f.with_columns(nsib=pl.len().over("s", "qk"), nrec=pl.len().over("s"), nsame=(pl.col("qk") == pl.col("sk")).sum().over("s"))
f.select("q", "s", "qk", "sk", "nsib", "nrec", "nsame").write_parquet(f"{OUT}/fr_keys.parquet")
y = f.filter(pl.col("pat").is_in(["swap|nsame", "added|nsame", "other|nsame", "dropped|nsame"]))
print("France: by nsib")
print(y.group_by("pat", pl.col("nsib").clip(1, 4)).agg(n=pl.len(), acc=pl.col("acc").mean(), rest=(pl.col("in7m") & ~pl.col("acc")).sum(),
                                                       p3=pl.col("p3").mean(), g=pl.col("p2g").mean()).sort("pat", "nsib"))
print("France same|nsame sib distribution", f.filter(pl.col("pat") == "same|nsame").group_by(pl.col("nsib").clip(1, 6)).len().sort("nsib"))
print("US same|nsame sib distribution", u.filter(pl.col("pat") == "same|nsame").group_by("country", pl.col("nsib").clip(1, 6)).len().sort("country", "nsib"))
