import polars as pl
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
F = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(42); pl.Config.set_tbl_width_chars(330)
top = pl.read_parquet(f"{T}/top_k3.parquet")
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
top = top.join(b.select("q", s_2="s", g_2="g", p3_2="p3"), on=["q", "s_2"], how="left")
h = top.filter(pl.col("s_2").is_not_null())
print("records with 2 candidates:", h.height, " accepted:", h["acc"].sum())
rk = {"same": 0, "added": 1, "dropped": 1, "squashed": 2, "acronym": 2, "swap": 3, "other": 5, "empty": 5}
h = h.with_columns(nk2=pl.col("pat_2").str.split("|").list.first(), num2=pl.col("pat_2").str.split("|").list.last())
h = h.with_columns(r1=pl.col("nk").replace_strict(rk, default=5), r2=pl.col("nk2").replace_strict(rk, default=5))
nr = {"nsame": 0, "nmiss": 1, "ndiff": 2}
h = h.with_columns(a1=pl.col("num").replace_strict(nr), a2=pl.col("num2").replace_strict(nr))
# 2nd candidate strictly better on name and not worse on number
bt = h.filter((pl.col("r2") < pl.col("r1")) & (pl.col("a2") <= pl.col("a1")))
print("2nd candidate strictly better name & not-worse number:", bt.height, " of which record accepted to 1st:", bt["acc"].sum())
print(bt.group_by("pat", "pat_2").agg(n=pl.len(), acc=pl.col("acc").sum(), p1=pl.col("p2").mean(), p_2=pl.col("p2_2").mean(), g1=pl.col("p2g").mean(), g2=pl.col("g_2").mean()).sort("n", descending=True).head(20))
bt.write_parquet(f"{T}/second_better.parquet")
print(bt.filter(pl.col("acc")).sample(15, seed=1).select("qn", "sn", "sn_2", "qa", "sa", "sa_2", "p2", "p2_2"))
