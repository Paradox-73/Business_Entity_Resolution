import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet("add_set_v2.parquet")
Ld, Lt = 0.0338, 0.0008
f = lambda x: x.agg(n=pl.len(), nlow=pl.col("low").sum(), p3=pl.col("p3").mean(), g=pl.col("p2g").mean()).with_columns(
    t=((Ld - pl.col("nlow")/pl.col("n"))/(Ld-Lt)).round(2), se=(((pl.col("nlow")/pl.col("n")).clip(1e-4)*(1-pl.col("nlow")/pl.col("n"))/pl.col("n")).sqrt()/(Ld-Lt)).round(2))
print(f(a.group_by("cls", pl.col("p2g").cut([0.1, 0.3, 0.5, 0.8]).alias("gb"))).sort("cls", "gb"))
print(f(a.group_by("cls", pl.col("p3").cut([0.01, 0.1, 0.3]).alias("pb"))).sort("cls", "pb"))
print(f(a.group_by("cls", ((pl.col("p2g") < 0.1) & (pl.col("p3") < 0.1)).alias("bothlow"))).sort("cls", "bothlow"))
