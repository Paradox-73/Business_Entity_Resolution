import sys, zlib, re, unicodedata, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import id_to_int
W = f"{WORK}/"
OUT = f'{SCRATCH}/final2/fr-strong-veto/'
STOP = set("rue r av ave avenue bd blvd boulevard ch chemin imp impasse all allee allée pl place cours quai route rte st saint ste de du des la le les l d et street road drive lane way court suite unit apt fl no n nº bis ter".split())
def words(a):
    if not a: return set()
    a = unicodedata.normalize('NFKD', a); a = ''.join(ch for ch in a if not unicodedata.combining(ch)).lower()
    parts = a.split(',')
    toks = set()
    for p in parts:
        if re.search(r'\d', p):  # the component with the house number = street line
            toks |= {t for t in re.findall(r"[a-z]{3,}", p) if t not in STOP}
    return toks
def repl(sa, qa):
    s, q = words(sa), words(qa)
    if not s or not q: return None
    return len(s & q) == 0
r = pl.read_parquet(OUT+'clean_subset.parquet')
p = pl.read_parquet(OUT+'fr_prof.parquet').filter(pl.col('grp')=='keep_cc')
ho = pl.read_parquet(OUT+'ho_prof.parquet').filter((pl.col('w')=='ab')&(pl.col('ob')<0.3)&(pl.col('gb')<0.3))
for nm, d in [('FR clean', r), ('FR keep', p), ('US/IN analog', ho)]:
    d = d.with_columns(rep=pl.Series([repl(a, b) for a, b in zip(d['sa'].to_list(), d['qa'].to_list())], dtype=pl.Boolean))
    if 'label' in d.columns:
        print(nm, d.group_by('label').agg(n=pl.len(), rep=pl.col('rep').mean(), nrep=pl.col('rep').sum()).rows())
    else:
        print(nm, d.height, 'street fully replaced share', d['rep'].mean(), 'n', d['rep'].sum())
    if nm == 'FR clean': d.write_parquet(OUT+'clean_subset_rep.parquet')
# US/IN true pairs sample (all pairs, not just candidates)
s1t = pl.read_parquet(W+"pairs/full/s1.parquet").sample(200000, seed=1)
s1t = s1t.with_columns(h=pl.col("s1_id").map_elements(lambda v: zlib.crc32(v.encode()) % 1000, return_dtype=pl.Int64)).filter(pl.col("h")<500).select("s","country")
g = pl.scan_parquet(W+"models/full_cons/oof.parquet").select("q","s","label").filter(pl.col("label")).collect().join(s1t, on="s").sample(20000, seed=2)
a1 = pl.scan_parquet(W+"train_s1.parquet").select(s=id_to_int("entity_id"), sa="business_address").join(g.select("s").unique().lazy(), on="s").collect()
aq = pl.concat([pl.scan_parquet(W+f"train_s{k}.parquet").select("entity_id","business_address") for k in (2,3)]).select(q=id_to_int("entity_id"), qa="business_address").join(g.select("q").unique().lazy(), on="q").collect()
g = g.join(a1, on="s").join(aq, on="q")
g = g.with_columns(rep=pl.Series([repl(a, b) for a, b in zip(g['sa'].to_list(), g['qa'].to_list())], dtype=pl.Boolean))
print('US/IN TRUE pairs sample', g.height, 'street fully replaced share', g['rep'].mean(), g.group_by('country').agg(pl.col('rep').mean()).rows())
