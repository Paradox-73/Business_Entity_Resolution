# Position of the noise word: appended at the END (after the legal form) vs in place. US/India T vs D, France refs and A1/A2.
import sys, re
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import words, undot_legal, LEGAL, NOISE
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
V = f'{SCRATCH}/frfix3/verify_recall/'
def shape(qn, sn):
    q = words(undot_legal(qn or '')); s = set(words(undot_legal(sn or '')))
    new = [i for i, w in enumerate(q) if w not in s and w in NOISE]
    if not new: return 'nonew'
    i = new[-1]
    leg = [j for j, w in enumerate(q) if w in LEGAL]
    last = i == len(q) - 1
    if leg and leg[0] < i and last: return 'end_after_legal'
    if last: return 'end'
    if i == 0: return 'first'
    return 'middle'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                   wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = u.filter(pl.col('wcls').is_in(['n_swap:noise', 'n_swap:desc>noise', 'n_add:noise']) & pl.col('num').is_in(['NSAME', 'NMISS']) | (pl.col('wcls').is_in(['n_swap:noise', 'n_add:noise']) & pl.col('num').str.starts_with('UP')))
u = u.join(pl.read_parquet(C + 'base.parquet', columns=['q', 'qn', 'sn']), on='q')
u = u.with_columns(shape=pl.Series([shape(a, b) for a, b in zip(u['qn'].to_list(), u['sn'].to_list())]), T=pl.col('grp') == 'T', numk=pl.col('num').str.replace(r'^UP.*', 'UP'))
t = u.group_by('wcls', 'numk', 'grp', 'shape').agg(n=pl.len()).with_columns(share=pl.col('n') / pl.col('n').sum().over('wcls', 'numk', 'grp'))
print(t.filter(pl.col('n') >= 5).sort('wcls', 'numk', 'grp', 'shape'))
# France
f = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fr_work.parquet', columns=['q', 's', 'qn', 'sn', 'key', 'acc9', 'lower', 'num2'])
P = {v: pl.read_parquet(V + f'fr_{v}.parquet') for v in ['v7ens', 'v9y']}
v9add = P['v9y'].join(P['v7ens'], on=['s', 'q'], how='anti')
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
fa = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fn_add_all.parquet', columns=['q', 's', 'tier', 'lower'])
S = {'FR acc SWAP noise NSAME (v7ens accepted)': f.filter(pl.col('acc9') & pl.col('key').is_in(['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME'])),
     'FR v9y 9.8k additions': f.join(v9add, on=['q', 's']),
     'FR A1 all (incl lowercase)': f.join(fa.filter(pl.col('tier') == 'A1').select('q', 's'), on=['q', 's']),
     'FR A2 all': f.join(fa.filter(pl.col('tier') == 'A2').select('q', 's'), on=['q', 's']),
     'FR rejected SWAP noise UP (decoy-like)': f.filter(~pl.col('acc9') & pl.col('key').is_in(['SWAP|n_swap:noise|UP', 'SWAP|n_swap:desc>noise|UP'])),
     'FR rejected ADD noise UP (decoy-like)': f.filter(~pl.col('acc9') & (pl.col('key') == 'ADD|n_add:noise|UP')),
     'FR rejected SWAP noise NSAME not in A1': f.filter(~pl.col('acc9') & pl.col('key').is_in(['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME'])).join(fa.select('q', 's'), on=['q', 's'], how='anti')}
for k, d in S.items():
    sh = pl.Series([shape(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())])
    d = d.with_columns(shape=sh)
    print(k, d.height, 'lower', round(d['lower'].mean(), 4), dict(zip(*[d['shape'].value_counts(normalize=True).sort('shape')[c].to_list() for c in ('shape', 'proportion')])))
    print('    lowercase by shape', d.group_by('shape').agg(n=pl.len(), low=pl.col('lower').mean()).sort('shape').rows())
print('US/India lowercase by shape and group (noise swap/add, NSAME + UP):')
print(u.with_columns(lower=pl.col('ops').list.contains('n_lower')).filter(pl.col('numk').is_in(['NSAME', 'UP'])).group_by('grp', 'shape').agg(n=pl.len(), low=pl.col('lower').mean()).filter(pl.col('n') >= 30).sort('grp', 'shape'))
