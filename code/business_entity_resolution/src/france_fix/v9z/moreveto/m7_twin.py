"""No-information names (brand words, acronyms): how many France S1 share the matched S1's address?"""
import sys, re, unicodedata
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(40)
OUT = f"{SCRATCH}/frfix3/moreveto"
s1 = pl.scan_parquet(f"{WORK}/test_s1.parquet").filter(pl.col("country") == "France").select(s=id_to_int("entity_id"), a="business_address").collect()
s1 = s1.with_columns(ak=pl.col("a").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]", ""))
s1 = s1.with_columns(k=pl.len().over("ak"))
g = pl.read_parquet(f"{OUT}/grp_table2.parquet").filter(pl.col("cur"))
h = pl.read_parquet(f"{OUT}/handle_chk.parquet", columns=["q", "s", "hk"])
g = g.join(h, on=["q", "s"], how="left").join(s1.select("s", "k"), on="s", how="left")
g = g.with_columns(info=pl.when(pl.col("qn").str.contains(r"^[A-Z]{2,4}$")).then(pl.lit("acronym"))
                   .when(pl.col("qn").str.contains(r"^[A-Z][a-z]{5,}$") & (pl.col("hk") == "mismatch")).then(pl.lit("brand"))
                   .when(pl.col("kind") == "other").then(pl.lit("other"))
                   .otherwise(pl.lit("named")))
print(g.group_by("info").agg(n=pl.len(), k1=(pl.col("k") == 1).mean(), k2=(pl.col("k") == 2).mean(), k3p=(pl.col("k") >= 3).mean(), kmean=pl.col("k").mean()))
print("France S1 address sharing: share of S1 with k>=2:", (s1["k"] >= 2).mean())
