import sys, re
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import DESC, words, STOPW, LEGAL
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
D = rf"{SCRATCH}/final/same-address/"
t = pl.read_parquet(D + "test_cands.parquet").filter(pl.col("country") == "France")
print("France same-key candidate pairs (unmatched records, after prefilter):", t.height, "records", t["q"].n_unique())
t = t.with_columns(nm=pl.col("nops").str.extract_all(r"n_[a-z_]+(:[a-z>]+)?").list.join("+"))
# descriptor swaps / adds at the SAME key (same house number + street words)
for pat in ["n_swap:desc", "n_add:desc", "n_drop:desc", "n_swap:", "n_add:", "n_drop:", "n_acronym"]:
    x = t.filter(pl.col("nops").str.contains(pat, literal=True))
    print(f"{pat:14s} pairs {x.height:6d} records {x['q'].n_unique():6d}  g_acc {x['g_acc'].mean():.3f} low {x['low'].mean():.3f}")
x = t.filter(pl.col("nops").str.contains("n_swap:desc", literal=True))
print(x.select("qn", "sn", "qa", "sa", "nops").sample(15, seed=1))
