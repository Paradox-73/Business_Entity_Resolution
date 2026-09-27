import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
RD = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/refute_desc"; NC = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"; FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
a = pl.read_parquet(f"{RD}/fr_low.parquet").join(pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "samestreet"]), on=["q", "s"], how="left")
Ld, Lt = 0.0338, 0.0008
x = a.filter(~pl.col("acc") & pl.col("kind").is_in(["swap", "added"]) & (pl.col("num") == "nmiss") & pl.col("cls").is_in(["N", "G", "rare"]))
print(x.group_by("cls", "samestreet").agg(n=pl.len(), nlow=pl.col("low").sum(), g=pl.col("p2g").mean(), p3=pl.col("p3").mean()).with_columns(t=((Ld-pl.col("nlow")/pl.col("n"))/(Ld-Lt)).round(2)).sort("cls", "samestreet"))
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "qn", "sn", "qa", "sa"])
print(x.filter(pl.col("cls") == "N").join(t, on=["q", "s"]).sample(20, seed=2).select("sn", "qn", "sa", "qa", "p3", "p2g"))
