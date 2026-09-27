import polars as pl, re
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(40); pl.Config.set_float_precision(4)
x = pl.read_parquet('fr_a1sig.parquet')
x = x.with_columns(src=pl.when(pl.col('acc9') & pl.col('gnadd')).then(pl.lit('acc9 via v9b gnadd')).when(pl.col('acc9')).then(pl.lit('acc9 (v7ens)')).otherwise(pl.col('state')))
f = pl.read_parquet('fn_dirs.parquet', columns=['q', 's', 'pair'])
x = x.join(pl.read_parquet('C:/ber_scratch/frfix2/census/fr_ops.parquet', columns=['q', 's', 'ops']), on=['q', 's'], how='left')
x = x.with_columns(hy=pl.col('ops').list.contains('n_hyphen'), s1hy=pl.col('sn').str.contains(r'\w-\w'),
                   dev=pl.col('qn').str.to_lowercase().str.contains('veloppement'), grp=pl.col('qn').str.to_lowercase().str.contains(r'\bgroupe\b'))
print(x.group_by('src').agg(n=pl.len(), low=pl.col('low').mean(), nlow=pl.col('low').sum(), hy_s1nohy=pl.col('hy').filter(~pl.col('s1hy')).mean(),
      dev=pl.col('dev').mean(), grp=pl.col('grp').mean(), g=pl.col('p2g').median(), t=pl.col('p3').median()).sort('src'))
# within 'developpement' records only
print(x.filter(pl.col('dev')).group_by('src').agg(n=pl.len(), low=pl.col('low').mean(), nlow=pl.col('low').sum(), hy=pl.col('hy').filter(~pl.col('s1hy')).mean(), g=pl.col('p2g').median(), t=pl.col('p3').median()).sort('src'))
print(x.filter(~pl.col('dev')).group_by('src').agg(n=pl.len(), low=pl.col('low').mean(), nlow=pl.col('low').sum(), hy=pl.col('hy').filter(~pl.col('s1hy')).mean(), g=pl.col('p2g').median(), t=pl.col('p3').median()).sort('src'))
x.write_parquet('fr_a1sig2.parquet')
