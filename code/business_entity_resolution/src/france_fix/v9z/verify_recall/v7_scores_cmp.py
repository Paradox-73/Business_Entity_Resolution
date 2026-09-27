# model scores (GBDT g, transformer p3, final p2) of the add set vs LB-tested sets (9,800 additions that worked; v7i, v7m adds that lost)
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(3)
V = f'{SCRATCH}/frfix3/verify_recall/'
R = f'{SCRATCH}/frfix3/recall/'
sc = pl.read_parquet(R + 'fr_scores.parquet', columns=['q', 's', 'p2', 'p3', 'g'])
P = {v: pl.read_parquet(V + f'fr_{v}.parquet') for v in ['v7ens', 'v7m', 'v7i', 'v9y']}
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
sets = {'v9y_additions(worked)': P['v9y'].join(P['v7ens'], on=['s', 'q'], how='anti'),
        'v7i_adds(LB~0.5)': P['v7i'].join(P['v7ens'], on=['s', 'q'], how='anti'),
        'v7m_adds(LB~0.4)': P['v7m'].join(P['v7ens'], on=['s', 'q'], how='anti')}
for g in add['grp'].unique().sort().to_list():
    sets['add:' + g] = add.filter(pl.col('grp') == g).select('s', 'q')
rows = []
for k, d in sets.items():
    x = d.join(sc, on=['q', 's'], how='left')
    rows.append(dict(set=k, n=x.height, g_med=x['g'].median(), g_lt05=(x['g'] < 0.5).mean(), p3_med=x['p3'].median(), p3_lt05=(x['p3'] < 0.5).mean(),
                     p3_lt01=(x['p3'] < 0.1).mean(), p2_med=x['p2'].median()))
print(pl.DataFrame(rows))
