# US/India: for the tier signatures, which extra ops (besides the key's own) occur on true vs false records?
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(4)
u = pl.read_parquet('usi_keys.parquet')
keys = ['SWAP|n_swap:noise|NSAME', 'TYPO||NSAME', 'SWAP|n_swap:desc|NSAME', 'SWAP|n_swap:other|NSAME', 'ADD|n_add:noise|NSAME', '||NSAME']
x = u.filter(pl.col('key').is_in(keys)).explode('ops').filter(pl.col('ops').is_not_null())
x = x.filter(~pl.col('ops').str.starts_with('n_swap') & ~pl.col('ops').str.starts_with('n_add') & ~pl.col('ops').str.starts_with('n_typo'))
tot = u.filter(pl.col('key').is_in(keys)).group_by('key', 'country', 'T').agg(N=pl.len())
g = x.group_by('key', 'country', 'T', 'ops').agg(n=pl.len()).join(tot, on=['key', 'country', 'T']).with_columns(r=pl.col('n') / pl.col('N'))
w = g.pivot(on='T', index=['key', 'country', 'ops'], values='r').rename({'true': 'rT', 'false': 'rF'}).fill_null(0)
for k in keys[:2]:
    print(k); print(w.filter((pl.col('key') == k)).sort('country', 'rF', descending=[False, True]).head(40))
print(tot.sort('key', 'country', 'T'))
