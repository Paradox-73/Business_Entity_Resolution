"""Add uppercase / other case ops as a second 'two changes' test; per group counts."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(50)
OUT = f"{SCRATCH}/frfix3/moreveto"
g = pl.read_parquet(f"{OUT}/grp_table.parquet")
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q", "s", "ops"])
g = g.join(o, on=["q", "s"], how="left")
g = g.with_columns(upx=pl.col("ops").list.contains("n_upper") & ~pl.col("handle"),
                   accx=pl.col("ops").list.contains("n_acc_strip") | pl.col("ops").list.contains("n_acc_add"),
                   brk=pl.col("ops").list.contains("n_bracket"), hyp=pl.col("ops").list.contains("n_hyphen"),
                   dbl=pl.col("ops").list.contains("n_dblspace"), leet=pl.col("ops").list.contains("n_leet"),
                   typo=pl.col("ops").list.eval(pl.element().str.starts_with("n_typo")).list.any())
g.drop("ops").write_parquet(f"{OUT}/grp_table2.parquet")
P = g.filter(pl.col("pop"))
r = (P.group_by("grp", "npat").agg(n_pop=pl.len(), n_cur=pl.col("cur").sum(), low=pl.col("lowx").mean(), up=pl.col("upx").mean(),
        brk=pl.col("brk").mean(), hyp=pl.col("hyp").mean(), dbl=pl.col("dbl").mean(), leet=pl.col("leet").mean(), typo=pl.col("typo").mean())
     .sort("n_cur", descending=True))
print(r.filter(pl.col("n_pop") >= 40))
V = g.filter(pl.col("dv") & pl.col("acc"))
print("descriptor veto set (fake reference):", V.select(pl.len(), pl.col("lowx").mean(), pl.col("upx").mean(), pl.col("brk").mean(), pl.col("hyp").mean(), pl.col("dbl").mean(), pl.col("leet").mean(), pl.col("typo").mean()))
