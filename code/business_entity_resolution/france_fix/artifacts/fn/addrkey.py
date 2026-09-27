# number of S1 rows sharing the S1 address key (house number + street words + first place component), per S1
import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from common import WORK, id_to_int
from fr_restore import street
OUT = 'C:/ber_scratch/frfix2/fn/'
def key(a):
    n, st = street(a)
    if n is None:
        return None
    comps = [c.strip().lower() for c in (a or '').split(',')]
    city = comps[1] if len(comps) > 1 else ''
    return f'{n}|{st}|{city}'
for split in ['test', 'train']:
    d = pl.read_parquet(f'{WORK}/{split}_s1.parquet', columns=['entity_id', 'business_address', 'country'])
    if split == 'test':
        d = d.filter(pl.col('country') == 'France')
    else:
        d = d.filter(pl.col('country') != 'France')
    k = [key(a) for a in d['business_address'].to_list()]
    d = d.select(s=id_to_int('entity_id')).with_columns(akey=pl.Series(k))
    d = d.with_columns(nakey=pl.len().over('akey'))
    d.write_parquet(OUT + f'akey_{split}.parquet')
    print(split, d.height, d['nakey'].value_counts().sort('nakey').head(6))
