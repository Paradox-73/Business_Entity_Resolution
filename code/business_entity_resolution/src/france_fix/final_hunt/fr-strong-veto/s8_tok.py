import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import id_to_int
W = f"{WORK}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
n = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase()
j = pl.read_parquet(OUT+"fr_v10c_scored.parquet")
rec = pl.concat([pl.scan_parquet(W+f"test_s{k}.parquet").select("entity_id","business_name") for k in (2,3)]).select(q=id_to_int("entity_id"), qn="business_name").join(j.select("q").lazy(), on="q").collect()
s1 = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id"), sn="business_name").collect()
x = j.join(rec, on="q").join(s1, on="s").with_columns(qq=n(pl.col("qn")), ss=n(pl.col("sn")))
cand = (pl.col("occ") & pl.col("gcc") & (pl.col("ob")<0.5) & (pl.col("gb")<0.5))
for t in ["developpement","groupe","associes","& fils","cie"]:
    y = x.filter(pl.col("qq").str.contains(t, literal=True) & ~pl.col("ss").str.contains(t, literal=True))
    print(f"{t:15s} v10c pairs {y.height:6d}  in cc(both) {y.filter(pl.col('occ')&pl.col('gcc')).height:6d}  cand {y.filter(cand).height:6d}  ob mean {y['ob'].mean()}  gb mean {y['gb'].mean()}  S1 names with token {s1.filter(n(pl.col('sn')).str.contains(t, literal=True)).height}")
# US/India held-out: English suffix tokens added to record, true pairs, bge p3
import zlib
s1t = pl.read_parquet(W+"pairs/full/s1.parquet")
s1t = s1t.with_columns(h=pl.col("s1_id").map_elements(lambda v: zlib.crc32(v.encode()) % 1000, return_dtype=pl.Int64)).filter(pl.col("h")<500).select("s","country")
ob = pl.read_parquet(W+"ce_b2/oof_s3_bgefolds.parquet", columns=["q","s","p3","label"]).join(s1t, on="s")
gb = pl.read_parquet(W+"gathik/v9/ce_x_oof_s3_bgef0bgef1bgef2.parquet", columns=["q","s","p3"]).rename({"p3":"gb"})
ob = ob.join(gb, on=["q","s"], how="inner")
r = pl.concat([pl.scan_parquet(W+f"train_s{k}.parquet").select("entity_id","business_name") for k in (2,3)]).select(q=id_to_int("entity_id"), qn="business_name").join(ob.select("q").unique().lazy(), on="q").collect()
ss = pl.scan_parquet(W+"train_s1.parquet").select(s=id_to_int("entity_id"), sn="business_name").join(ob.select("s").unique().lazy(), on="s").collect()
y = ob.join(r, on="q").join(ss, on="s").with_columns(qq=n(pl.col("qn")), sq=n(pl.col("sn")))
for t in ["development","group","associates","& sons","& co"," co","holdings","enterprises","services"]:
    z = y.filter(pl.col("qq").str.contains(t, literal=True) & ~pl.col("sq").str.contains(t, literal=True))
    zt = z.filter(pl.col("label"))
    print(f"US/IN {t:12s} cc pairs {z.height:6d} true {z['label'].mean() if z.height else 0:.3f} | true pairs: ob mean {zt['p3'].mean()} gb mean {zt['gb'].mean()} share both<0.5 {((zt['p3']<0.5)&(zt['gb']<0.5)).mean() if zt.height else 0}")
