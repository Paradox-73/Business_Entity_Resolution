import polars as pl
pl.Config.set_tbl_rows(150); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(62); pl.Config.set_float_precision(3)
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_work.parquet')
na = d.filter((pl.col('num2') == 'NMISS') & (pl.col('sm') == 'na') & (pl.col('stw') >= 0.99))
print('NMISS with no number anywhere and all S1 street words present:')
print(na.group_by('key').agg(n=pl.len(), acc=pl.col('acc9').mean(), rej=(~pl.col('acc9')).sum(), la=pl.col('lower').filter(pl.col('acc9')).mean(),
      lr=pl.col('lower').filter(~pl.col('acc9')).mean(), nlr=pl.col('lower').filter(~pl.col('acc9')).sum(), g=pl.col('p2g').filter(~pl.col('acc9')).median(),
      t=pl.col('p3').filter(~pl.col('acc9')).median(), st2=(pl.col('sm2') == 'st').filter(~pl.col('acc9')).mean()).filter(pl.col('rej') >= 40).sort('rej', descending=True).head(40))
st = d.filter((pl.col('num2') == 'NSAME') & (pl.col('sm') == 'st') & ~pl.col('acc9'))
print('same street, rejected, split by GBDT p2g >= 0.5:')
print(st.filter(pl.col('key').is_in(['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME', 'TYPO||NSAME', 'SWAP|n_swap:other|NSAME'])).group_by('key', pl.col('p2g') >= 0.5, pl.col('p3') >= 0.5).agg(
    n=pl.len(), lr=pl.col('lower').mean(), nl=pl.col('lower').sum()).sort('key', 'p2g', 'p3'))
