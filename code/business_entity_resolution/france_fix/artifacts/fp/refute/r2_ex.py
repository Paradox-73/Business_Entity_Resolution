"""Examples: US/India true records with LEGAL_ADD|UP and SWAP|UP; France tier A examples."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
T = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/'
u = pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'country', 'p', 'base', 'numc', 'num', 'low'])
x = u.filter((pl.col('base') == 'LEGAL_ADD') & (pl.col('numc') == 'UP'))
top = pl.read_parquet(T + 'france/usi_top.parquet', columns=['q', 'qn', 'qa', 'sn', 'sa'])
x = x.join(top, on='q', how='left')
print('== US/India TRUE LEGAL_ADD|UP examples')
print(x.filter(pl.col('grp') == 'T').sample(25, seed=1).select('country', 'p', 'num', 'qn', 'sn', 'qa', 'sa'))
print('== US/India num detail for true vs false LEGAL_ADD|UP')
print(x.group_by('grp', 'num').len().sort('grp', 'len', descending=True))
print('== US FALSE LEGAL_ADD|UP with p>=0.5 examples')
print(x.filter((pl.col('grp') != 'T') & (pl.col('p') >= 0.5)).head(15).select('country', 'grp', 'p', 'num', 'qn', 'sn', 'qa', 'sa'))
v = pl.read_parquet(F + 'fp_veto_set.parquet')
print('== France veto tier A examples')
print(v.filter(pl.col('tier') == 'A').sample(20, seed=2).select('p2', 'num', 'low', 'qn', 'sn', 'qa', 'sa'))
print(v.group_by('tier').agg(n=pl.len(), p2mean=pl.col('p2').mean(), p2min=pl.col('p2').min(), p2q=pl.col('p2').quantile(0.25), low=pl.col('low').sum()))
print(v.filter(pl.col('tier') == 'A').group_by('num').len().sort('len', descending=True))
