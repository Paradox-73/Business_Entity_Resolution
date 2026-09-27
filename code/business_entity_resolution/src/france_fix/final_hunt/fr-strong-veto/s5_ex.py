import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
c = pl.read_parquet(OUT+"fr_prof.parquet").filter(pl.col("grp")=="cand")
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(260)
x = c.filter(pl.col("ob")<0.1).sample(40, seed=3)
print(x.select("sn","qn","sa","qa", pl.col("ob").round(3), pl.col("gb").round(3), pl.col("og").round(2),"nm","num","n_v","ntw"))
