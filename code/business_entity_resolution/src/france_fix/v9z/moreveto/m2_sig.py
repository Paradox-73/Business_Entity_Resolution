"""Accepted v9y France pairs by name-change signature: size, lowercase share, implied true rate."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
OUT = f"{SCRATCH}/frfix3/moreveto"
v = pl.read_parquet(f"{OUT}/acc_v9y.parquet")
v = v.with_columns(low=pl.col("nc").list.contains("LOWER"),
                   sig=pl.col("nc").list.filter(pl.element() != "LOWER").list.join("+"))
TL, DL = 0.0030, 0.0338
g = (v.group_by("sig", "num").agg(n=pl.len(), low=pl.col("low").mean(), nlow=pl.col("low").sum(), p2=pl.col("p2").mean())
     .with_columns(t_est=((DL - pl.col("low")) / (DL - TL)).clip(0, 1.2)).sort("n", descending=True))
print(g.filter(pl.col("n") >= 150).head(80))
