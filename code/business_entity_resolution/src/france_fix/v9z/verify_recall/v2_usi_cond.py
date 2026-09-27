# US/India truth rate of the add-set signatures conditional on the model rejecting / scoring low (selection effect)
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(lower=pl.col('ops').list.contains('n_lower'), num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
                   bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                   wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = u.with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'), T=pl.col('grp') == 'T')
keys = ['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME', 'SWAP|n_swap:noise|NMISS', 'SWAP|n_swap:desc>noise|NMISS', 'ADD|n_add:noise|NMISS',
        'ADD+SWAP|n_add:noise n_swap:noise|NMISS', 'LEGAL_DROP||NMISS', 'LEGAL_ADD||NMISS', 'TYPO||NMISS', 'TYPO||NSAME',
        'ACRONYM||NSAME', 'ACRONYM+LEGAL_DROP||NSAME', 'ACRONYM||NMISS', 'ACRONYM+LEGAL_DROP||NMISS', 'DOMAIN||NSAME', 'DOMAIN||NMISS', 'LEGAL_DOT||NSAME', 'LEGAL_DOT||NMISS']
x = u.filter(pl.col('key').is_in(keys) & ~pl.col('lower'))
x = x.with_columns(band=pl.when(pl.col('p') < 0.05).then(pl.lit('a<0.05')).when(pl.col('p') < 0.3).then(pl.lit('b.05-.3')).when(pl.col('p') < 0.5).then(pl.lit('c.3-.5'))
                   .when(pl.col('p') < 0.8).then(pl.lit('d.5-.8')).otherwise(pl.lit('e>=.8')))
t = x.group_by('key', 'band').agg(n=pl.len(), tr=pl.col('T').mean(), nD=(pl.col('grp') == 'D').sum(), nW=(pl.col('grp') == 'W').sum()).sort('key', 'band')
print(t)
print(x.group_by('key').agg(n=pl.len(), tr=pl.col('T').mean(), low_p=(pl.col('p') < 0.8).sum(), tr_lowp=pl.col('T').filter(pl.col('p') < 0.8).mean(),
      tr_p5=pl.col('T').filter(pl.col('p') < 0.5).mean()).sort('key'))
