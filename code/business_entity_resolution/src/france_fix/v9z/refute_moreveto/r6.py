import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/fp"))
from fp1_dec import france_scores
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(55)
NEW = f"{WORK}/frfix3/test_scores_v9y_moreveto_stem.parquet"
V = pl.read_parquet(f"{WORK}/frfix3/moreveto_stem_set.parquet")
S = [10354232030, 10659305782]
print("added S1 rows in veto set:", V.filter(pl.col("s").is_in(S)).select("q","s","sn","qn","p2").to_dicts())
b = france_scores(NEW)
b = b.sort("p2", descending=True).unique("q", keep="first").filter(pl.col("s").is_in(S) & (pl.col("p2") >= 0.3))
t = pl.read_parquet(f"{SCRATCH}/france/fr_top.parquet", columns=["q","s","qn","qa","sn","sa"])
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q","s","nc","num"])
print(b.join(t, on=["q","s"], how="left").join(o, on=["q","s"], how="left").sort("s","p2", descending=[False,True]).select("s","q","p2","qn","sn","qa","sa","nc","num"))
