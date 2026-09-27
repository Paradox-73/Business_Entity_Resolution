import os, sys
import polars as pl
T = r"C:/ber_scratch/final/fr-committee"
c = pl.read_parquet(os.path.join(T, "cand.parquet")).filter(pl.col("kind") == "add")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
c = c.with_columns(lowok=~pl.col("ops").str.contains("n_domain|n_squash"),
                   word=pl.col("ops").str.contains("n_swap:|n_add:|n_drop:"),
                   legal=pl.col("ops").str.contains("n_legal_change|n_legal_add"),
                   up=pl.col("ops").str.contains("a_num_up"), down=pl.col("ops").str.contains("a_num_down"),
                   pob=pl.col("po").cut([0.1, 0.3, 0.5, 0.7, 0.9]), pgb=pl.col("pg").cut([0.5, 0.7, 0.8, 0.9, 0.95, 0.99]),
                   own=pl.col("s") == pl.col("s_best"))
agg = lambda d, k: d.group_by(k).agg(n=pl.len(), nlowok=pl.col("lowok").sum(), low=pl.col("low").filter(pl.col("lowok")).mean(),
                                   up=pl.col("up").mean(), down=pl.col("down").mean(), po=pl.col("po").mean(), pg=pl.col("pg").mean(),
                                   empty=(pl.col("n_v") == 0).mean()).sort("n", descending=True)
print("ALL", c.height, "lowok low", c.filter(pl.col("lowok"))["low"].mean())
print(agg(c, ["word", "legal"]))
print(agg(c, "pob").sort("pob"))
print(agg(c, "pgb").sort("pgb"))
print(c.group_by("own").agg(n=pl.len(), po=pl.col("po").mean(), pg=pl.col("pg").mean()))
print(agg(c, "nm").head(40))
print(agg(c, "num"))
print(c.select("po_best").describe())
print(c.filter(~pl.col("own")).select("s", "s_best", "po", "po_best", "pg").head(10))
