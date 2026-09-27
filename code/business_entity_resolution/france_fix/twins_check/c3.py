import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
T = r"C:/ber_scratch/frfix/twins"
F = r"C:/ber_scratch/france"
s1 = pl.read_parquet(f"{T}/fr_s1kk.parquet")
s1n = pl.read_parquet(f"{T}/fr_s1.parquet")
s1 = s1.join(s1n, on="s")
s1 = s1.with_columns(dk=pl.concat_str([pl.col("ks"), pl.col("fs"), pl.col("snum").fill_null("-"), pl.col("sst"), pl.col("scity")], separator="|"))
s1 = s1.with_columns(gs=pl.len().over("dk"), gsn=pl.len().over("ks", "scity"))
print("S1 in exact duplicate groups (name+form+num+street+city):", (s1["gs"] >= 2).sum())
print("S1 sharing name key within city:", (s1["gsn"] >= 2).sum(), " dist:", s1["gsn"].clip(1, 6).value_counts().sort("gsn"))
d = s1.filter(pl.col("gs") >= 2).sort("dk")
print(d.head(10).select("sn", "sa", "gs"))
top = pl.read_parquet(f"{T}/top_k3.parquet", columns=["q", "s", "s_2", "p2", "p2_2", "acc", "pat", "pat_2", "p2g"])
dd = d.select("s", "dk")
x = top.join(dd, on="s").join(dd.rename({"s": "s_2", "dk": "dk2"}), on="s_2", how="left")
print("records whose best S1 is in an exact-dup S1 group:", x.height, " accepted:", x["acc"].sum(), " 2nd is same group:", (x["dk"] == x["dk2"]).sum())
y = x.filter(pl.col("dk") == pl.col("dk2"))
print(y.select("p2", "p2_2", "p2g").describe())
