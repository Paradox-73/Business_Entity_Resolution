import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(3)
OUT = 'C:/ber_scratch/frfix2/fn/'
tok = r'^\s*[A-Za-z]{5,}\s*$'
d = pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'qn', 'sn', 'qa', 'sa', 'p2g', 'p3', 'p9', 'acc9', 'p2_2', 'num', 'nc'])
d = d.join(pl.read_parquet(OUT + 'fr_street.parquet'), on=['q', 's']).join(pl.read_parquet(OUT + 'ovl_fr.parquet'), on=['q', 's'])
d = d.join(pl.read_parquet(OUT + 'akey_test.parquet'), on='s', how='left').with_columns(uniq=(pl.col('nakey') == 1) & pl.col('akey').is_not_null())
d = d.with_columns(inv=pl.col('qn').str.contains(tok) & (pl.col('ovm') == 0),
                   case=pl.when(pl.col('qn') == pl.col('qn').str.to_lowercase()).then(pl.lit('lower')).when(pl.col('qn') == pl.col('qn').str.to_uppercase()).then(pl.lit('upper')).otherwise(pl.lit('mixed')))
x = d.filter(pl.col('inv'))
print(x.group_by('sm', 'uniq').agg(n=pl.len(), acc=pl.col('acc9').mean(), rej=(~pl.col('acc9')).sum(), low_a=(pl.col('case') == 'lower').filter(pl.col('acc9')).mean(),
      low_r=(pl.col('case') == 'lower').filter(~pl.col('acc9')).mean(), up_a=(pl.col('case') == 'upper').filter(pl.col('acc9')).mean(), up_r=(pl.col('case') == 'upper').filter(~pl.col('acc9')).mean(),
      p2g_r=pl.col('p2g').filter(~pl.col('acc9')).median(), p3_r=pl.col('p3').filter(~pl.col('acc9')).median(), tw_r=(pl.col('p2_2') > 0.1).filter(~pl.col('acc9')).mean(),
      st2_r=(pl.col('sm2') == 'st').filter(~pl.col('acc9')).mean()).sort('n', descending=True))
y = x.filter((pl.col('sm') == 'st') & pl.col('uniq') & ~pl.col('acc9'))
print(y.sample(15, seed=2).select('qn', 'sn', 'qa', 'sa', 'p2g', 'p3', 'p2_2', 'sm2'))
d.select('q', 's', 'inv', 'uniq', 'nakey', 'case', 'ovm', 'nqw', 'nsw').write_parquet(OUT + 'fr_extra.parquet')
