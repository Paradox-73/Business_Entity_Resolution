# France: address/format op rates in decoy reference (descriptor veto set, LB-confirmed ~0 true), true references
# (accepted noise-word swap/add at same number; the 9,800 v9y additions, LB-supported) and each add-set tier.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(330); pl.Config.set_fmt_str_lengths(40); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
V = f'{SCRATCH}/frfix3/verify_recall/'
f = pl.read_parquet(C + 'fr_ops.parquet', columns=['q', 's', 'acc', 'ops', 'nc', 'num'])
f = f.with_columns(bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'))
veto = pl.read_parquet(f'{SCRATCH}/frfix/namechg/veto_set.parquet', columns=['q', 's'])
P = {v: pl.read_parquet(V + f'fr_{v}.parquet') for v in ['v7ens', 'v9y', 'v7j', 'v7i', 'v7m']}
v9add = P['v9y'].join(P['v7ens'], on=['s', 'q'], how='anti')
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
acc_ok = P['v9y'].join(v9add, on=['s', 'q'], how='anti')   # kept from v7ens
wc = pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add):noise'))).list.len() > 0
G = {}
G['DEC veto set'] = f.join(veto, on=['q', 's'])
G['DEC v7j adds'] = f.join(P['v7j'].join(P['v7ens'], on=['s','q'], how='anti'), on=['q', 's'])
G['TRUE v9y 9.8k adds'] = f.join(v9add, on=['q', 's'])
G['TRUE acc noise NSAME'] = f.join(acc_ok, on=['q', 's']).filter((pl.col('num') == 'NSAME') & wc & pl.col('bnc').is_in(['SWAP', 'ADD']))
G['TRUE acc nochange NSAME'] = f.join(acc_ok, on=['q', 's']).filter((pl.col('num') == 'NSAME') & (pl.col('bnc') == ''))
G['TRUE acc NMISS'] = f.join(acc_ok, on=['q', 's']).filter(pl.col('num') == 'NMISS')
G['mix v7i adds'] = f.join(P['v7i'].join(P['v7ens'], on=['s','q'], how='anti'), on=['q', 's'])
G['mix v7m adds'] = f.join(P['v7m'].join(P['v7ens'], on=['s','q'], how='anti'), on=['q', 's'])
for g in add['grp'].unique().sort().to_list():
    G['add ' + g[:24]] = f.join(add.filter(pl.col('grp') == g).select('q', 's'), on=['q', 's'])
OPS = ['a_street_abbr', 'a_street_expand', 'a_state_expand', 'a_state_abbr', 'a_chg:street', 'a_acc_add', 'a_acc_strip', 'a_drop:place', 'a_drop:state', 'a_num_suffix', 'a_num_dot', 'a_upper', 'a_lower', 'a_shuffle', 'a_typo', 'a_num_pad', 'a_num_prefix', 'n_lower', 'n_upper', 'n_acc_strip', 'n_acc_add']
rows = []
for k, d in G.items():
    r = {'set': k, 'n': d.height}
    for o in OPS:
        r[o.replace('a_', '').replace('n_', 'N')] = d['ops'].list.contains(o).mean() if d.height else None
    rows.append(r)
T = pl.DataFrame(rows)
print(T)
T.write_parquet(V + 'fr_art_rates.parquet')
