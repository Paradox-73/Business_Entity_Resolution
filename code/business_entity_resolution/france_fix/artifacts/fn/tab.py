import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
d = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/fr_base.parquet', columns=['q', 's', 'p2', 'p9', 'p2g', 'p3', 'acc9', 'nc', 'num', 'ops', 'p2_2'])
d = d.with_columns(num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
                   lower=pl.col('ops').list.contains('n_lower'))
d = d.with_columns(sig2=pl.col('nc').list.join('+') + '|' + pl.col('num2'))
u = pl.read_csv('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census/signatures.csv')
u = u.group_by('sig2').agg(
    tr_us=(pl.col('T').filter(pl.col('country') == 'US').sum() / pl.col('n').filter(pl.col('country') == 'US').sum()),
    n_us=pl.col('n').filter(pl.col('country') == 'US').sum(),
    tr_in=(pl.col('T').filter(pl.col('country') == 'India').sum() / pl.col('n').filter(pl.col('country') == 'India').sum()),
    n_in=pl.col('n').filter(pl.col('country') == 'India').sum())
g = d.group_by('sig2').agg(n=pl.len(), acc=pl.col('acc9').mean(), rej=(~pl.col('acc9')).sum(),
                           low_acc=pl.col('lower').filter(pl.col('acc9')).mean(), low_rej=pl.col('lower').filter(~pl.col('acc9')).mean(),
                           p9_rej=pl.col('p9').filter(~pl.col('acc9')).median(), p2g_rej=pl.col('p2g').filter(~pl.col('acc9')).median(),
                           tw_rej=(pl.col('p2_2') > 0.3).filter(~pl.col('acc9')).mean())
g = g.join(u, on='sig2', how='left').sort('rej', descending=True)
g.write_csv('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/fr_rej_sigs.csv')
print(g.filter(~pl.col('sig2').str.contains(r'\|(UP|DOWN)$')).head(70))
