"""France: keep rate per added word (all best candidates with that word added, swap/added patterns); decompose restored/accepted."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
OUT = f"{SCRATCH}/frfix/namechg"
c = pl.read_parquet(f"{OUT}/fr_chg.parquet").drop("qa", "sa", "sn_2")
c = c.with_columns(num=pl.col("pat").str.split("|").list.last(), nk=pl.col("pat").str.split("|").list.first())
x = c.filter(pl.col("nk").is_in(["swap", "added"]) & (pl.col("added").list.len() == 1)).with_columns(a=pl.col("added").list.first())
t = x.group_by("a").agg(n=pl.len(), keep=(pl.col("num") == "nsame").sum() / (pl.col("num") != "nmiss").sum(),
                        n_same=(pl.col("num") == "nsame").sum(), n_swap=(pl.col("nk") == "swap").sum(),
                        acc_same=pl.col("acc").filter(pl.col("num") == "nsame").mean(),
                        acc_n=pl.col("acc").filter(pl.col("num") == "nsame").sum(),
                        rest_n=(pl.col("in7m") & ~pl.col("acc")).filter(pl.col("num") == "nsame").sum(),
                        p3_same=pl.col("p3").filter(pl.col("num") == "nsame").mean(),
                        g_same=pl.col("p2g").filter(pl.col("num") == "nsame").mean())
t = t.sort("n", descending=True)
print(t.head(110))
t.write_parquet(f"{OUT}/fr_word_keep.parquet")
import numpy as np
k = t.filter(pl.col("n") >= 200)["keep"].to_numpy()
print("keep-rate histogram (words with n>=200):", np.histogram(k, bins=np.arange(0, 1.05, 0.05)))
