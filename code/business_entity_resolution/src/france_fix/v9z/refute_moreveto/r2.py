import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(60)
NC = f"{SCRATCH}/frfix/namechg"
c = pl.read_parquet(f"{NC}/fr_chg.parquet", columns=["q", "s", "acc", "p2", "pat", "added", "dropped", "qn", "sn", "twin", "twin_samenum", "ns1"])
o = pl.read_parquet(f"{SCRATCH}/frfix2/census/fr_ops.parquet", columns=["q", "s", "nc", "num"])
c = c.join(o, on=["q", "s"], how="left")
print(c["num"].value_counts().sort("count", descending=True))
w = pl.read_parquet(f"{NC}/word_class.parquet")
wc = dict(zip(w["a"].to_list(), w["cls"].to_list()))
print("word classes:", w["cls"].value_counts())
for x in ["sport","sportive","sportif","college","collectif","groupe","groupement","compagnie","cie","association","amicale"]:
    print(x, w.filter(pl.col("a")==x).to_dicts())
c = c.with_columns(nA=pl.col("added").list.len(), nD=pl.col("dropped").list.len())
one = c.filter((pl.col("nA") == 1) & (pl.col("nD") == 1)).with_columns(a=pl.col("added").list.first(), d=pl.col("dropped").list.first())
one = one.with_columns(ca=pl.col("a").replace_strict(wc, default="rare"), cd=pl.col("d").replace_strict(wc, default="rare"))
one = one.with_columns(stem=(pl.col("a").str.slice(0,5)==pl.col("d").str.slice(0,5)) & (pl.col("a").str.len_chars()>=5) & (pl.col("d").str.len_chars()>=5))
one.write_parquet(f"{SCRATCH}/frfix3/refute_moreveto/one_swaps.parquet")
# replacement distribution for dropped sport / college / groupe / sportive / collectif
for d in ["sport", "sportive", "college", "collectif", "groupe"]:
    z = one.filter(pl.col("d") == d)
    print(f"--- dropped {d}: {z.height} single swaps; by added class:", z.group_by("ca").len().sort("len", descending=True).to_dicts())
    print(z.filter(pl.col("ca")=="D").group_by("a").agg(n=pl.len(), acc=pl.col("acc").sum()).sort("n", descending=True).head(12).to_dicts())
