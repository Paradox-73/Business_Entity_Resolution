import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import id_to_int
W = f"{WORK}/"
OUT = f'{SCRATCH}/final2/fr-strong-veto/'
src = open(OUT+'s17_street.py', encoding='utf8').read().split("r = pl.read_parquet")[0]
exec(src)
norm = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase().str.replace_all(r"[^a-z0-9]+"," ").str.strip_chars()
d = pl.read_parquet(OUT+'clean_subset_rep.parquet')
s1 = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s2=id_to_int("entity_id"), k=norm(pl.col("business_name")), sa2="business_address").collect()
d = d.with_columns(k=norm(pl.col("sn")))
tw = d.select("s","q","k","qa").join(s1, on="k").filter(pl.col("s2")!=pl.col("s"))
tw = tw.with_columns(hit=pl.Series([len(words(a) & words(b))>0 for a, b in zip(tw['sa2'].to_list(), tw['qa'].to_list())], dtype=pl.Boolean))
h = tw.group_by("s","q").agg(twin_street_hit=pl.col("hit").any(), ntw_other=pl.len())
d = d.join(h, on=["s","q"], how="left").with_columns(pl.col("twin_street_hit").fill_null(False))
for f, nm in [(pl.col('rep')==True, 'street replaced'), (pl.col('rep')!=True, 'not replaced/unknown')]:
    x = d.filter(f.fill_null(False)) if nm=='street replaced' else d.filter(~(pl.col('rep').fill_null(False)))
    print(nm, x.height, 'twin S1 matches record street:', x['twin_street_hit'].sum(), 'has other twin:', x['ntw_other'].is_not_null().sum())
x = d.filter(pl.col('rep').fill_null(False) & pl.col('twin_street_hit'))
print(x.select('sn','sa','qn','qa',pl.col('ob').round(3),pl.col('gb').round(3)).head(10))
d.write_parquet(OUT+'clean_subset_rep.parquet')
