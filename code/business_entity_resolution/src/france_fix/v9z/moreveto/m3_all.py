"""All France best candidates by signature: lowercase share, acceptance (v9y) of lowercase vs not, model scores."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
OUT = f"{SCRATCH}/frfix3/moreveto"
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q", "s", "p2", "p2g", "nc", "num"])
v = pl.read_parquet(f"{OUT}/acc_v9y.parquet", columns=["q", "s"]).with_columns(a9=pl.lit(True))
o = o.join(v, on=["q", "s"], how="left").with_columns(pl.col("a9").fill_null(False))
o = o.with_columns(low=pl.col("nc").list.contains("LOWER"), sig=pl.col("nc").list.filter(pl.element() != "LOWER").list.join("+"))
o.select("q", "s", "sig", "num", "low", "a9", "p2", "p2g").write_parquet(f"{OUT}/all_sig.parquet")
g = (o.group_by("sig", "num").agg(n=pl.len(), low=pl.col("low").mean(), acc=pl.col("a9").mean(),
        acc_low=pl.col("a9").filter(pl.col("low")).mean(), acc_nl=pl.col("a9").filter(~pl.col("low")).mean(),
        g_low=pl.col("p2g").filter(pl.col("low")).mean(), g_nl=pl.col("p2g").filter(~pl.col("low")).mean(),
        p_low=pl.col("p2").filter(pl.col("low")).mean(), p_nl=pl.col("p2").filter(~pl.col("low")).mean())
     .sort("n", descending=True))
print(g.head(45))
