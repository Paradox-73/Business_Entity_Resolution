"""Verify the proposed scores file vs v9b (rows, p1, p2 only changed on France veto pairs) and the built TSV vs v9b TSV."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
W2 = f'{WORK}/frfix2/'
old = pl.read_parquet(os.path.join(WORK, 'frfix', 'test_scores_frmin_descveto_gnadd.parquet'))
new = pl.read_parquet(W2 + 'test_scores_v9b_fpveto.parquet')
print('cols old', old.columns, old.height, 'new', new.columns, new.height)
s1 = pl.read_parquet(os.path.join(WORK, 'test_s1.parquet'), columns=['entity_id', 'country']).select(s=id_to_int('entity_id'), country='country')
j = old.join(new, on=['q', 's'], how='full', suffix='_n', coalesce=True)
print('rows only in old', j.filter(pl.col('p2_n').is_null()).height, 'only in new', j.filter(pl.col('p2').is_null()).height)
j = j.join(s1, on='s', how='left')
d = j.filter((pl.col('p2') != pl.col('p2_n')) | (pl.col('p1') != pl.col('p1_n')))
print('changed rows', d.height, 'by country', d.group_by('country').len().to_dicts(), 'p1 changed', j.filter(pl.col('p1') != pl.col('p1_n')).height)
print('changed rows new p2 values', d['p2_n'].unique().to_list()[:5], 'old p2 min/max', d['p2'].min(), d['p2'].max())
v = pl.read_parquet(W2 + 'fp_veto_set.parquet', columns=['q', 's', 'tier'])
print('veto set', v.height, 'unique', v.select('q', 's').n_unique(), 'changed rows in veto', d.join(v, on=['q', 's']).height)
print('scores dtypes', new.schema)
del old, new, j
def tsv(p):
    t = pl.read_csv(p, separator='\t', schema_overrides={'matched_entity_ids': pl.Utf8})
    return (t.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids')
             .filter(pl.col('matched_entity_ids') != '').select(s=id_to_int('source1_entity_id'), q=id_to_int('matched_entity_ids')))
a = tsv(os.path.join(WORK, 'frfix', 'out_descveto_gnadd_all', 'matching_results.tsv')).join(s1, on='s')
b = tsv(W2 + 'refute_fp_out/matching_results.tsv').join(s1, on='s')
for c in ['France', 'US', 'India']:
    x, y = a.filter(pl.col('country') == c), b.filter(pl.col('country') == c)
    print(c, 'v9b', x.height, 'new', y.height, 'removed', x.join(y, on=['s', 'q'], how='anti').height, 'added', y.join(x, on=['s', 'q'], how='anti').height)
rem = a.join(b, on=['s', 'q'], how='anti')
add = b.join(a, on=['s', 'q'], how='anti')
print('removed in veto', rem.join(v, on=['s', 'q']).height)
sc = pl.read_parquet(W2 + 'test_scores_v9b_fpveto.parquet', columns=['q', 's', 'p2'])
print('removed not in veto:', rem.join(v, on=['s', 'q'], how='anti').join(sc, on=['q', 's'], how='left').to_dicts())
print('added:', add.join(sc, on=['q', 's'], how='left').sort('p2').to_dicts())
rem.write_parquet(f'{SCRATCH}/frfix2/fp/refute/removed.parquet')
add.write_parquet(f'{SCRATCH}/frfix2/fp/refute/added.parquet')
