import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
import polars as pl
from common import WORK, id_to_int
a = pl.read_parquet(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet')
b = pl.read_parquet(f'{WORK}/frfix3/test_scores_v9y_recall.parquet')
print('rows', a.height, b.height, 'schema', a.schema, b.schema)
print('dups b', b.select('q','s').is_duplicated().sum())
j = b.join(a, on=['q','s'], how='left', suffix='_a')
miss = j.filter(pl.col('p2_a').is_null())
print('rows in b not in a', miss.height); print(miss)
print('rows in a not in b', a.join(b, on=['q','s'], how='anti').height)
ch = j.filter(pl.col('p2_a').is_not_null() & ((pl.col('p2') != pl.col('p2_a')) | (pl.col('p1') != pl.col('p1_a'))))
print('changed rows', ch.height, 'p1 changed', (ch['p1'] != ch['p1_a']).sum())
print(ch.select(pl.col('p2_a').describe() if False else pl.col('p2_a')).describe())
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id','country']).select(s=id_to_int('entity_id'), c=pl.col('country'))
x = ch.join(s1, on='s', how='left')
print(x['c'].value_counts())
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
print('add', add.height, 'uniq q', add['q'].n_unique(), 'uniq s', add['s'].n_unique(), add['grp'].value_counts().sort('grp'))
chq = pl.concat([ch.select('q','s'), miss.select('q','s')])
print('changed == add set?', chq.join(add, on=['q','s']).height, chq.height)
# which records do the q's belong to (country of record)
q23 = pl.concat([pl.read_parquet(f'{WORK}/test_s{k}.parquet', columns=['entity_id','country']) for k in (2,3)]).select(q=id_to_int('entity_id'), qc=pl.col('country'))
print(add.join(q23, on='q', how='left')['qc'].value_counts())
print(add.join(s1, on='s', how='left')['c'].value_counts())
# are the added pairs p2_a distribution
print(add.join(a, on=['q','s'], how='left').group_by('grp').agg(n=pl.len(), nnull=pl.col('p2').is_null().sum(), p2med=pl.col('p2').median(), p2max=pl.col('p2').max()).sort('grp'))
