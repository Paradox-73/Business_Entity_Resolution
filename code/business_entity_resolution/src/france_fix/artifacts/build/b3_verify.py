import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
import sys
import polars as pl
from common import WORK, id_to_int
R = f'{ROOT}/'
W = R + 'work/frfix2/'
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).rename({'entity_id': 'source1_entity_id'})
def rd(p):
    return pl.read_csv(p, separator='\t', schema_overrides={'source1_entity_id': pl.Utf8, 'matched_entity_ids': pl.Utf8}, quote_char=None).with_columns(pl.col('matched_entity_ids').fill_null(''))
def pairs(p):
    d = rd(p).join(s1, on='source1_entity_id').filter(pl.col('country') == 'France')
    return d.with_columns(q=pl.col('matched_entity_ids').str.split(',')).explode('q').filter(pl.col('q') != '').select(s1='source1_entity_id', q='q')
def rowdiff(a, b):
    x = rd(a).join(rd(b), on='source1_entity_id', suffix='_b', how='full', coalesce=True).filter(pl.col('matched_entity_ids') != pl.col('matched_entity_ids_b')).join(s1, on='source1_entity_id', how='left')
    return x['country'].value_counts().sort('country').rows()
base = pairs(R + 'work/frfix/out_descveto_gnadd_all/matching_results.tsv')
print('v9b France pairs', base.height)
veto = pl.read_parquet(W + 'fp_veto_set.parquet', columns=['q', 's', 'tier'])
fnall = pl.read_parquet(W + 'fn_add_set_final.parquet')
tomap = lambda d: d.with_columns(q=pl.col('q').cast(pl.Utf8), s=pl.col('s').cast(pl.Utf8))
for v, add in [('fpfn', fnall), ('fp', None)]:
    P = pairs(W + f'out_{v}_all/matching_results.tsv')
    rem = base.join(P, on=['s1', 'q'], how='anti'); ad = P.join(base, on=['s1', 'q'], how='anti')
    s1chg = pl.concat([rem['s1'], ad['s1']]).n_unique()
    # knock-on: removals that are not veto pairs, additions that are not fn adds
    # ids in TSV are the raw entity ids; map veto/add (int) to raw by id_to_int on the TSV side
    remi = rem.with_columns(s=id_to_int('s1'), qi=id_to_int('q'))
    adi = ad.with_columns(s=id_to_int('s1'), qi=id_to_int('q'))
    rv = remi.join(veto.rename({'q': 'qi'}), on=['s', 'qi'], how='inner')
    print(f'--- {v}: France pairs {P.height}; removed {rem.height} (veto pairs {rv.height}, by tier {rv["tier"].value_counts().sort("tier").rows()}; knock-on {rem.height - rv.height}); added {ad.height}', end='')
    if add is not None:
        aa = adi.join(add.rename({'q': 'qi'}), on=['s', 'qi'], how='inner').height
        miss = add.height - aa
        print(f' (fn adds {aa} of {add.height}, not accepted {miss}; knock-on {ad.height - aa})', end='')
    print(f'; S1 rows changed {s1chg}')
    for m, v9 in [('v7ens', 'v9b'), ('v7p', 'v9d')]:
        out = W + f'out_{v}_on_{m}/matching_results.tsv'
        print(f'  {out}: vs main {m}', rowdiff(out, R + f'submissions/{m}/matching_results.tsv'), f'| vs {v9}', rowdiff(out, R + f'submissions/{v9}/matching_results.tsv'))
