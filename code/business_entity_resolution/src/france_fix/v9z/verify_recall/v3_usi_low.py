# Does the lowercase test separate US/India decoys from true pairs INSIDE the add-set signatures (incl. low-p ones)?
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
C = f'{SCRATCH}/frfix2/census/'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(lower=pl.col('ops').list.contains('n_lower'), num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
                   bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                   wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = u.with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'), T=pl.col('grp') == 'T')
keys = ['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME', 'SWAP|n_swap:noise|NMISS', 'ADD|n_add:noise|NMISS', 'LEGAL_DROP||NMISS', 'LEGAL_ADD||NMISS', 'TYPO||NMISS', 'TYPO||NSAME', 'ADD|n_add:noise|NSAME', 'SWAP|n_swap:other|NSAME', 'SWAP|n_swap:desc|NSAME', 'ADD|n_add:other|NSAME', 'LEGAL_ADD||NSAME', 'LEGAL_ADD||UP', 'ADD|n_add:noise|UP', 'SWAP|n_swap:noise|UP']
x = u.filter(pl.col('key').is_in(keys))
print(x.group_by('key', pl.col('grp')).agg(n=pl.len(), low=pl.col('lower').mean(), low_plo=pl.col('lower').filter(pl.col('p') < 0.3).mean(), n_plo=(pl.col('p') < 0.3).sum()).sort('key', 'grp'))
# overall reference: all T with exactly one content name change vs all D
print('T one-change lowercase', u.filter(pl.col('T') & (pl.col('bnc') != '') & ~pl.col('bnc').str.contains(r'\+'))['lower'].mean(),
      '| T no content change', u.filter(pl.col('T') & (pl.col('bnc') == ''))['lower'].mean(), '| D all', u.filter(pl.col('grp') == 'D')['lower'].mean(),
      '| W all', u.filter(pl.col('grp') == 'W')['lower'].mean())
