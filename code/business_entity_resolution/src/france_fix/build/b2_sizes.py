import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
RD = f"{SCRATCH}/frfix/refute_desc"
NC = f"{SCRATCH}/frfix/namechg"
a = pl.read_parquet(f"{RD}/fr_low.parquet").join(pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "samestreet", "p2"]), on=["q", "s"], how="left")
Ld, Lt = 0.0338, 0.0008
def est(x):
    return x.agg(n=pl.len(), nlow=pl.col("low").sum(), g=pl.col("p2g").mean().round(3), p3=pl.col("p3").mean().round(3)).with_columns(
        t=((Ld - pl.col("nlow")/pl.col("n")) / (Ld - Lt)).round(2),
        se=((((pl.col("nlow")/pl.col("n")).clip(1e-4) * (1 - pl.col("nlow")/pl.col("n")) / pl.col("n")).sqrt()) / (Ld - Lt)).round(2))
sa = a.filter(pl.col("kind").is_in(["swap", "added"]))
R = sa.filter(~pl.col("acc") & (pl.col("num") == "nsame") & pl.col("cls").is_in(["N", "G"]))
print("rejected N/G swap/added nsame"); print(est(R.group_by("cls", "kind", "in7m", "samestreet")).sort("cls", "kind", "in7m", "samestreet"))
print("v7m restored (in7m & ~acc) by cls/kind (all nums)")
print(est(a.filter(pl.col("in7m") & ~pl.col("acc")).group_by("cls", "kind")).sort("n", descending=True))
print("accepted swap/added groups n>=100 with t<0.7")
print(est(sa.filter(pl.col("acc")).group_by("cls", "kind", "num")).filter((pl.col("n") >= 100)).sort("t"))
