import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250)
T = r"C:/ber_scratch/frfix/twins"
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet")
print(s1.select("sa", "scity", "snum", "sst").sample(15, seed=4))
print("S1 city empty:", (s1["scity"] == "").mean(), " distinct cities:", s1["scity"].n_unique())
print(s1["scity"].value_counts().sort("count", descending=True).head(15))
r = pl.read_parquet(f"{T}/fr_recsk.parquet", columns=["qa", "qcity", "qnum", "qst"])
print(r.sample(15, seed=4))
print("rec city empty:", (r["qcity"] == "").mean(), " distinct:", r["qcity"].n_unique())
