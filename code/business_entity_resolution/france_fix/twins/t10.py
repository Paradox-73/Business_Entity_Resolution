import polars as pl
T = r"C:/ber_scratch/frfix/twins"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(320)
top = pl.read_parquet(f"{T}/top_k2.parquet")
f = top.filter(pl.col("nk").is_in(["swap", "added", "other"]))
print(f.group_by("nk", "num").agg(n=pl.len(), ex_all=(pl.col("nx_all") > 0).mean(), ex_city=(pl.col("nx_city") > 0).mean(), n_excity=(pl.col("nx_city") > 0).sum(),
      acc_n_excity=(pl.col("acc") & (pl.col("nx_city") > 0)).sum()).sort("n", descending=True))
x = f.filter((pl.col("pat") == "swap|nsame") & (pl.col("nx_city") > 0) & pl.col("acc"))
print("swap|nsame accepted with exact-name S1 in same city:", x.height, " name multiplicity nx_city distribution:")
print(x["nx_city"].value_counts().sort("nx_city").head(10))
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet")
print(x.sample(25, seed=2).select("qn", "sn", "qa", "sa", "nx_city", "p2", "p2g", "p3"))
