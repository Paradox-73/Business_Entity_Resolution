import polars as pl, sys, re, unicodedata
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from fr_restore import street
T = r"C:/ber_scratch/frfix/twins"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(330)
u = pl.read_parquet(f"{T}/us_top_k.parquet", columns=["q", "s", "p", "label", "pat", "kq", "qcity", "qa", "qn", "sn", "c"])
st = [street(a) for a in u["qa"].to_list()]
u = u.with_columns(qnum=pl.Series([x[0] for x in st], dtype=pl.Utf8), qst=pl.Series([x[1] for x in st]))
u = u.with_columns(dk=pl.concat_str([pl.col("kq"), pl.col("qnum").fill_null("-"), pl.col("qst"), pl.col("qcity")], separator="|"), acc=pl.col("p") >= 0.5)
u = u.with_columns(gsz=pl.len().over("dk"), nacc=pl.col("acc").sum().over("dk"))
d = u.filter(pl.col("gsz") >= 2)
print("US records in duplicate groups:", d.height)
mixed = d.filter((pl.col("nacc") > 0) & (pl.col("nacc") < pl.col("gsz")))
accS = mixed.filter(pl.col("acc")).select("dk", sacc="s").unique()
un = mixed.filter(~pl.col("acc")).join(accS, on="dk")
same = un.filter(pl.col("s") == pl.col("sacc")).unique("q")
print("US mixed groups: unaccepted members whose best S1 == accepted S1 of the group:", same.height, " truth rate:", same["label"].mean(), " mean p:", same["p"].mean())
print(same.group_by(pb=pl.col("p").cut([0.1, 0.3, 0.5])).agg(n=pl.len(), rate=pl.col("label").mean(), pm=pl.col("p").mean()).sort("pb"))
g = d.filter(pl.col("acc")).group_by("dk").agg(nS=pl.col("s").n_unique(), n=pl.len())
diff = d.filter(pl.col("acc")).join(g.filter(pl.col("nS") > 1), on="dk")
print("US duplicates accepted to different S1:", diff.height, " truth rate:", diff["label"].mean(), " mean p:", diff["p"].mean())
