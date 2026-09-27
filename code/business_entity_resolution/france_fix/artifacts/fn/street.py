# street match for France best candidates: same first house number AND same street words (fr_restore.street + fuzzy)
import sys, time
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from rapidfuzz import fuzz
from fr_restore import street
OUT = 'C:/ber_scratch/frfix2/fn/'
t = time.time()
d = pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'qa', 'sa', 's_2', 'sa_2'])
def sm(qa, sa):
    a, b = street(sa), street(qa)
    if a[0] is None or b[0] is None:
        return 'na', 0
    r = fuzz.token_sort_ratio(a[1], b[1]) if a[1] and b[1] else 0
    r2 = fuzz.token_set_ratio(a[1], b[1]) if a[1] and b[1] else 0
    r = max(r, r2)
    if a[0] != b[0]:
        return ('nd_st' if r >= 85 else 'nd_dst'), r
    return ('st' if r >= 85 else 'dst'), r
res = [sm(x, y) for x, y in zip(d['qa'].to_list(), d['sa'].to_list())]
res2 = [sm(x, y) if y is not None else ('none', 0) for x, y in zip(d['qa'].to_list(), d['sa_2'].to_list())]
out = d.select('q', 's').with_columns(sm=pl.Series([r[0] for r in res]), sr=pl.Series([int(r[1]) for r in res], dtype=pl.Int16),
                                      sm2=pl.Series([r[0] for r in res2]))
out.write_parquet(OUT + 'fr_street.parquet')
print(out['sm'].value_counts(), out['sm2'].value_counts(), round(time.time() - t))
