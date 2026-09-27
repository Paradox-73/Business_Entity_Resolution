import polars as pl, re
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(4)
C = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census/'
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
# single invented word: one alphabetic token, >=5 letters, no digits
tok = r'^\s*[A-Za-z]{5,}\s*$'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num'])
v = pl.read_parquet(OUT + 'ovl_usi.parquet')
b = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france/usi_top.parquet', columns=['q', 'qn', 'sn'])
u = u.join(v, on='q').join(b, on='q').with_columns(num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'))
u = u.with_columns(inv=pl.col('qn').str.contains(tok) & (pl.col('ovm') == 0))
u = u.with_columns(case=pl.when(pl.col('qn') == pl.col('qn').str.to_lowercase()).then(pl.lit('lower')).when(pl.col('qn') == pl.col('qn').str.to_uppercase()).then(pl.lit('upper')).otherwise(pl.lit('mixed')))
print(u.filter(pl.col('inv')).group_by('country', 'num2').agg(n=pl.len(), T=(pl.col('grp') == 'T').sum(), D=(pl.col('grp') == 'D').sum(), W=(pl.col('grp') == 'W').sum(),
      lowT=(pl.col('case') == 'lower').filter(pl.col('grp') == 'T').mean(), lowD=(pl.col('case') == 'lower').filter(pl.col('grp') != 'T').mean(),
      upT=(pl.col('case') == 'upper').filter(pl.col('grp') == 'T').mean(), accT=(pl.col('p') >= 0.5).filter(pl.col('grp') == 'T').mean(),
      accF=(pl.col('p') >= 0.5).filter(pl.col('grp') != 'T').mean()).sort('country', 'n', descending=[False, True]))
print(u.filter(pl.col('inv') & (pl.col('grp') == 'D') & (pl.col('num2') == 'NSAME')).sample(10, seed=1).select('qn', 'sn', 'p'))
print(u.filter(pl.col('inv') & (pl.col('grp') == 'T')).sample(15, seed=1).select('qn', 'sn', 'p'))
print('==== by S1 address-key count')
ak = pl.read_parquet(OUT + 'akey_train.parquet')
s = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france/usi_top.parquet', columns=['q', 's'])
u = u.join(s, on='q').join(ak, on='s', how='left').with_columns(uniq=(pl.col('nakey') == 1) & pl.col('akey').is_not_null())
print(u.filter(pl.col('inv') & (pl.col('num2') == 'NSAME')).group_by('country', 'uniq').agg(n=pl.len(), T=(pl.col('grp') == 'T').sum(), D=(pl.col('grp') == 'D').sum(), W=(pl.col('grp') == 'W').sum(),
      tr=(pl.col('grp') == 'T').mean(), lowT=(pl.col('case') == 'lower').filter(pl.col('grp') == 'T').mean(), lowF=(pl.col('case') == 'lower').filter(pl.col('grp') != 'T').mean(),
      accT=(pl.col('p') >= 0.5).filter(pl.col('grp') == 'T').mean(), accF=(pl.col('p') >= 0.5).filter(pl.col('grp') != 'T').mean()).sort('country', 'uniq'))
# same for all zero-overlap (any form) NSAME
print(u.filter((pl.col('ovm') == 0) & (pl.col('num2') == 'NSAME')).group_by('country', 'uniq', 'inv').agg(n=pl.len(), tr=(pl.col('grp') == 'T').mean(), W=(pl.col('grp') == 'W').mean()).sort('country', 'uniq', 'inv'))
