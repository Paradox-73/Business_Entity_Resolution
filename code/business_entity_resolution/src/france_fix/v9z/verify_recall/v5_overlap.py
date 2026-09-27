import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(4)
V = f'{SCRATCH}/frfix3/verify_recall/'
P = {v: pl.read_parquet(V + f'fr_{v}.parquet') for v in ['v7ens', 'v7m', 'v7i', 'v7j', 'v9b', 'v9e', 'v9f', 'v9y']}
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'business_name', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'), sn='business_name')
qq = pl.concat([pl.read_parquet(f'{WORK}/test_s{k}.parquet', columns=['entity_id', 'business_name', 'country']).filter(pl.col('country') == 'France') for k in (2, 3)]).select(q=id_to_int('entity_id'), qn='business_name')
def low(df):
    d = df.join(qq, on='q', how='left').join(s1, on='s', how='left')
    return d.with_columns(lower=(pl.col('qn').str.contains(r'\p{L}') & (pl.col('qn') == pl.col('qn').str.to_lowercase()) & (pl.col('sn') != pl.col('sn').str.to_lowercase())).fill_null(False))
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
base = P['v7ens']
rows = []
for v in ['v7m', 'v7i', 'v7j', 'v9b', 'v9e', 'v9f', 'v9y']:
    plus = P[v].join(base, on=['s', 'q'], how='anti'); minus = base.join(P[v], on=['s', 'q'], how='anti')
    lp, lm = low(plus), low(minus)
    ov = add.join(plus, on=['s', 'q'])
    ovq = add.join(plus.select('q'), on='q')     # same record added (maybe another S1)
    rows.append(dict(v=v, added=plus.height, add_low=lp['lower'].mean(), removed=minus.height, rem_low=lm['lower'].mean(), ovl_pairs=ov.height, ovl_rec=ovq.height))
    if ov.height:
        print(v, 'overlap by grp'); print(ov.group_by('grp').agg(n=pl.len()).sort('grp'))
    if v == 'v9y':
        # was any add-set record matched by v7ens (and removed)?
        print('add-set records matched in v7ens:', add.join(base.select('q', s0='s'), on='q').height, '| same S1', add.join(base, on=['q', 's']).height)
        print(add.join(base.select('q', s0='s'), on='q').join(add.join(base, on=['q','s']).select('q').with_columns(same=pl.lit(True)), on='q', how='left').group_by('grp').agg(n=pl.len(), same=pl.col('same').sum()).sort('grp'))
print(pl.DataFrame(rows))
# lowercase of the add set itself
la = low(add)
print('add-set lowercase (should be ~0; lowercase were filtered out):'); print(la.group_by('grp').agg(n=pl.len(), low=pl.col('lower').sum()).sort('grp'))
