# per-S1 match-count distribution: US/India train truth, US/India v9y predicted, France v9y, France v9y + adds
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int, read_truth
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(4)
V = f'{SCRATCH}/frfix3/verify_recall/'
gt = read_truth()
print(gt.columns, gt.height)
s1t = pl.read_parquet(f'{WORK}/train_s1.parquet', columns=['entity_id', 'country']).select(s=id_to_int('entity_id'), c='country')
gcol = [c for c in gt.columns]
def dist(pairs, s1, tag):
    k = s1.join(pairs.group_by('s').agg(k=pl.len()), on='s', how='left').with_columns(pl.col('k').fill_null(0).clip(0, 9))
    return k.group_by('k').agg(n=pl.len()).with_columns(share=pl.col('n') / pl.col('n').sum()).sort('k').select('k', pl.col('share').alias(tag))
try:
    g = gt.rename({gcol[0]: 's', gcol[1]: 'q'}) if 's' not in gcol else gt
    g = g.select(s=id_to_int('s'), q=id_to_int('q'))
except Exception as e:
    print(e); g = gt
out = None
for c in ['US', 'India']:
    d = dist(g, s1t.filter(pl.col('c') == c), 'truth_' + c)
    out = d if out is None else out.join(d, on='k', how='full', coalesce=True)
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).select(s=id_to_int('entity_id'), c='country')
fr = s1.filter(pl.col('c') == 'France')
v9 = pl.read_parquet(V + 'fr_v9y.parquet')
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
out = out.join(dist(v9, fr, 'FR_v9y'), on='k', how='full', coalesce=True).join(dist(pl.concat([v9, add.select('s', 'q')]), fr, 'FR_v9y+adds'), on='k', how='full', coalesce=True)
out = out.join(dist(pl.read_parquet(V + 'fr_v7ens.parquet'), fr, 'FR_v7ens'), on='k', how='full', coalesce=True)
print(out.sort('k'))
print('means: FR v9y', v9.height / fr.height, 'FR +adds', (v9.height + add.height) / fr.height)
