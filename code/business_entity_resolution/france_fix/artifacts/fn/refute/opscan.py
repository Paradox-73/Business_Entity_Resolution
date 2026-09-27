# which ops separate true from false within word-change keys at the same house number (US/India labels)?
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(4)
u = pl.read_parquet('usi_keys.parquet', columns=['key', 'country', 'T', 'ops'])
u = u.filter(pl.col('key').str.contains(r'^(SWAP|ADD|TYPO)[^|]*\|[^|]*\|NSAME$'))
tot = u.group_by('country', 'T').agg(N=pl.len())
x = u.explode('ops').filter(pl.col('ops').is_not_null()).filter(~pl.col('ops').str.contains(r'^n_(swap|add|drop|typo)'))
g = x.group_by('country', 'T', 'ops').agg(n=pl.len()).join(tot, on=['country', 'T']).with_columns(r=pl.col('n') / pl.col('N'))
w = g.pivot(on='T', index=['country', 'ops'], values='r').rename({'true': 'rT', 'false': 'rF'}).fill_null(0)
w = w.with_columns(ratio=(pl.col('rF') + 1e-4) / (pl.col('rT') + 1e-4))
print(tot)
print(w.filter((pl.col('rF') >= 0.01) | (pl.col('rT') >= 0.01)).sort('ratio', descending=True).head(30))
print(w.filter((pl.col('rF') >= 0.01) | (pl.col('rT') >= 0.01)).sort('ratio').head(12))
