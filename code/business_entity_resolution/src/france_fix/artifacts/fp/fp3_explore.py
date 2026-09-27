import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
F = f'{SCRATCH}/frfix2/fp/'
u = pl.read_parquet(F + 'usi_sig.parquet')
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'acc', 'acc9', 'veto', 'gadd', 'numc', 'low', 'base', 'nchg', 'bsig'])

print('== numc distribution by group/country (share)')
x = u.group_by('country', 'grp', 'numc').len().with_columns(sh=pl.col('len') / pl.col('len').sum().over('country', 'grp'))
print(x.pivot(on='numc', index=['country', 'grp'], values='sh').sort('country', 'grp'))
print('France all top pairs numc share, by v9b acc')
x = fr.group_by('acc9', 'numc').len().with_columns(sh=pl.col('len') / pl.col('len').sum().over('acc9'))
print(x.pivot(on='numc', index='acc9', values='len'))

print('== lowercase share by nchg (0,1,2,3+) group/country')
print(u.with_columns(k=pl.col('nchg').clip(0, 3)).group_by('country', 'grp', 'k').agg(n=pl.len(), low=pl.col('low').mean())
       .pivot(on='k', index=['country', 'grp'], values='low', sort_columns=True).sort('country', 'grp'))
print('France acc9 lowercase by nchg')
print(fr.with_columns(k=pl.col('nchg').clip(0, 3)).group_by('acc9', 'veto', 'k').agg(n=pl.len(), low=pl.col('low').mean()).sort('acc9', 'veto', 'k'))

# US/India true rates per (bsig, low)
def rates(d, keys):
    return d.group_by(keys).agg(n=pl.len(), T=(pl.col('grp') == 'T').sum(), D=(pl.col('grp') == 'D').sum(),
                                W=(pl.col('grp') == 'W').sum()).with_columns(tr=pl.col('T') / pl.col('n'))

ru = rates(u.filter(pl.col('country') == 'US'), ['bsig', 'low']).rename({'n': 'n_us', 'tr': 'tr_us'}).select('bsig', 'low', 'n_us', 'tr_us')
ri = rates(u.filter(pl.col('country') == 'India'), ['bsig', 'low']).rename({'n': 'n_in', 'tr': 'tr_in'}).select('bsig', 'low', 'n_in', 'tr_in')
ra = rates(u, ['bsig', 'low']).rename({'n': 'n_ui', 'tr': 'tr_ui'}).select('bsig', 'low', 'n_ui', 'tr_ui')
g = (fr.filter(pl.col('acc9')).group_by('bsig', 'low').agg(n_acc=pl.len(), p2=pl.col('p2').mean())
       .join(ru, on=['bsig', 'low'], how='left').join(ri, on=['bsig', 'low'], how='left').join(ra, on=['bsig', 'low'], how='left'))
g.write_parquet(F + 'acc_sig_rates.parquet')
print('== accepted v9b France pairs by (bsig, low), US/India true rates; sorted by n_acc, tr_ui < 0.6')
print(g.filter(pl.col('tr_ui') < 0.6).sort('n_acc', descending=True).head(60))
print('total accepted with tr_ui<0.6:', g.filter(pl.col('tr_ui') < 0.6)['n_acc'].sum(), ' <0.76:',
      g.filter(pl.col('tr_ui') < 0.76)['n_acc'].sum(), ' no ref:', g.filter(pl.col('tr_ui').is_null())['n_acc'].sum())
print('== biggest accepted groups overall')
print(g.sort('n_acc', descending=True).head(40))
