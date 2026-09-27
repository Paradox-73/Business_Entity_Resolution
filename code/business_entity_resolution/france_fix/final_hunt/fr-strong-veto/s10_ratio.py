import sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import id_to_int
W = "E:/Projects/Amazon ML Challenge/work/"
OUT = "C:/ber_scratch/final2/fr-strong-veto/"
n = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase()
def cnt(split, country, toks):
    s1 = pl.scan_parquet(W+f"{split}_s1.parquet").filter(pl.col("country")==country).select(x=n(pl.col("business_name")))
    rc = pl.concat([pl.scan_parquet(W+f"{split}_s{k}.parquet").filter(pl.col("country")==country).select(x=n(pl.col("business_name"))) for k in (2,3)])
    a = s1.select([pl.len().alias("N")]+[pl.col("x").str.contains(t, literal=True).sum().alias(t) for t in toks]).collect()
    b = rc.select([pl.len().alias("N")]+[pl.col("x").str.contains(t, literal=True).sum().alias(t) for t in toks]).collect()
    print(split, country, "S1:", a.row(0, named=True)); print(split, country, "REC:", b.row(0, named=True))
cnt("test","France",["developpement","groupe","associes","& fils"," cie","amicale","ecole","club"])
cnt("train","US",["group","associates","& sons","development","holdings","partners","services"])
cnt("test","US",["group","associates","& sons","development","holdings","partners","services"])
