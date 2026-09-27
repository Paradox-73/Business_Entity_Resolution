import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(48); pl.Config.set_tbl_width_chars(330)
bt = pl.read_parquet(f"{T}/second_better.parquet")
ft = pl.read_parquet(f"{F}/fr_top.parquet", columns=["q", "sn_2", "sa_2"])
x = bt.filter((pl.col("pat") == "swap|nsame") & (pl.col("pat_2") == "same|nsame")).join(ft, on="q")
print(x.filter(pl.col("acc")).sample(20, seed=1).select("qn", "sn", "sn_2", "qa", "sa", "sa_2", "p2", "p2_2"))
print(x.filter(~pl.col("acc")).sample(8, seed=1).select("qn", "sn", "sn_2", "qa", "sa", "sa_2", "p2", "p2_2"))
