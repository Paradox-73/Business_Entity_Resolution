import os
import polars as pl
T = r"C:/ber_scratch/final/fr-committee"
a = pl.read_parquet(os.path.join(T, "train_committee.parquet")).filter(pl.col("kind") == "add")
a = a.with_columns(amiss=pl.col("ops").str.contains("a_missing"), pgb=pl.col("pg").cut([0.8, 0.9, 0.95, 0.99]), pob=pl.col("po").cut([0.2, 0.4, 0.5]),
                   emp=pl.col("n_v") == 0)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40)
ag = lambda d, k: d.group_by(k).agg(n=pl.len(), prec=pl.col("y").mean().round(3), lowT=pl.col("low").filter(pl.col("lowok") & pl.col("y")).mean().round(3),
                                    lowF=pl.col("low").filter(pl.col("lowok") & ~pl.col("y")).mean().round(3),
                                    upT=pl.col("up").filter(pl.col("y")).mean().round(3), upF=pl.col("up").filter(~pl.col("y")).mean().round(3),
                                    dnT=pl.col("down").filter(pl.col("y")).mean().round(3), dnF=pl.col("down").filter(~pl.col("y")).mean().round(3)).sort(k)
print(ag(a, ["cls"]))
print(ag(a, ["country", "cls"]))
print(ag(a, ["cls", "amiss"]))
print(ag(a, ["cls", "emp"]))
print(ag(a, ["cls", "pgb"]))
print(ag(a, ["cls", "pob"]))
print(ag(a.filter(pl.col("cls") == "noword"), "num"))
nw = a.filter((pl.col("cls") == "noword") & ~pl.col("up"))
print("noword, no num up:", nw.height, nw["y"].mean())
print(ag(nw, "pgb"))
