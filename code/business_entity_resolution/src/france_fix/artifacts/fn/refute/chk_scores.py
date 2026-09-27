import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
from common import WORK  # noqa: E402
import polars as pl
from common import WORK, id_to_int
old = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet')
new = pl.read_parquet(f'{WORK}/frfix2/test_scores_v9b_fnadd.parquet')
print('schemas', old.schema, new.schema)
print('rows', old.height, new.height)
same_order = (old['q'] == new['q']).all() and (old['s'] == new['s']).all()
print('same (q,s) row order:', same_order)
print('p1 identical:', (old['p1'] == new['p1']).all() if 'p1' in old.columns else 'n/a')
print('unique (q,s) new:', new.select('q', 's').n_unique())
diff = new.with_columns(p2o=old['p2']).filter(pl.col('p2') != pl.col('p2o'))
print('pairs with changed p2:', diff.height, '| new p2 values:', diff['p2'].unique().to_list()[:5], '| old p2 range', diff['p2o'].min(), diff['p2o'].max())
add = pl.read_parquet(f'{WORK}/frfix2/fn_add_set.parquet')
print('add set', add.height, add.select('q', 's').n_unique(), '| changed pairs in add set:', diff.join(add, on=['q', 's']).height,
      '| add-set pairs already 0.95+ in v9b:', new.join(add, on=['q', 's']).with_columns(p2o=pl.lit(0)).height)
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).select(s=id_to_int('entity_id'), country='country')
print('changed pairs by S1 country:', diff.join(s1, on='s', how='left')['country'].value_counts())
print('add-set pairs missing from scores:', add.join(new.select('q', 's'), on=['q', 's'], how='anti').height)
