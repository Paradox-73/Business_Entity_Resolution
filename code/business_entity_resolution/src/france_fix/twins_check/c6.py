import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
top = pl.read_parquet(f"{F}/fr_top.parquet", columns=["q", "s", "p2", "p2g", "acc", "pat", "s_2", "p2_2", "pat_2"])
b = pl.read_parquet(f"{T}/fr_pairs.parquet", columns=["q", "s", "g"])
x = top.filter((pl.col("pat") == "swap|nsame") & (pl.col("pat_2") == "same|nsame")).join(b.rename({"s": "s_2", "g": "g2"}), on=["q", "s_2"], how="left")
print("swap|nsame 1st, same|nsame 2nd:", x.height, " accepted to 1st:", x["acc"].sum(), " mean p 1st", x["p2"].mean(), " 2nd", x["p2_2"].mean(), " g1", x["p2g"].mean(), " g2", x["g2"].mean())
print("any record with 2nd p2_2 >= 0.5:", (top["p2_2"] >= 0.5).sum(), "; France S1 accepted count mean:", top.filter("acc").height / 259452)
