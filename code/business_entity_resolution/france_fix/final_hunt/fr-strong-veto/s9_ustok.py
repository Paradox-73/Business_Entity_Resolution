import sys, zlib, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import id_to_int
W = "E:/Projects/Amazon ML Challenge/work/"
n = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase()
toks = ["group","associates","& sons","development","holdings","enterprises","partners","international"]
pat = "|".join(t.replace("&", "&") for t in toks)
r = (pl.concat([pl.scan_parquet(W+f"train_s{k}.parquet").select("entity_id","business_name") for k in (2,3)])
     .select(q=id_to_int("entity_id"), qq=n(pl.col("business_name"))).filter(pl.col("qq").str.contains(pat)).collect())
print("train records with tokens", r.height)
s1t = pl.read_parquet(W+"pairs/full/s1.parquet")
s1t = s1t.with_columns(h=pl.col("s1_id").map_elements(lambda v: zlib.crc32(v.encode()) % 1000, return_dtype=pl.Int64)).filter(pl.col("h")<500).select("s","country")
g = pl.scan_parquet(W+"models/full_cons/oof.parquet").select("q","s","p2","label").join(r.lazy(), on="q").collect().join(s1t, on="s")
ss = pl.scan_parquet(W+"train_s1.parquet").select(s=id_to_int("entity_id"), sq=n(pl.col("business_name"))).join(g.select("s").unique().lazy(), on="s").collect()
g = g.join(ss, on="s")
ob = pl.read_parquet(W+"ce_b2/oof_s3_bgefolds.parquet", columns=["q","s","p3"])
gb = pl.read_parquet(W+"gathik/v9/ce_x_oof_s3_bgef0bgef1bgef2.parquet", columns=["q","s","p3"]).rename({"p3":"gb"})
g = g.join(ob, on=["q","s"], how="left").join(gb, on=["q","s"], how="left").with_columns(pl.col("label").fill_null(False))
for t in toks:
    z = g.filter(pl.col("qq").str.contains(t, literal=True) & ~pl.col("sq").str.contains(t, literal=True))
    zt = z.filter(pl.col("label"))
    zc = zt.filter(pl.col("p3").is_not_null() & pl.col("gb").is_not_null())
    print(f"{t:14s} all oof pairs {z.height:7d} true {zt.height:6d} | true in cc(both) {zc.height:5d} both<0.5 {((zc['p3']<0.5)&(zc['gb']<0.5)).sum():4d} | GBDT p2>=0.5 pairs {z.filter(pl.col('p2')>=0.5).height} true share {z.filter(pl.col('p2')>=0.5)['label'].mean()}")
