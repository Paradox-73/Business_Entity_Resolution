import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(3)
R = f'{SCRATCH}/frfix3/recall/'
V = f'{SCRATCH}/frfix3/verify_recall/'
c = pl.read_parquet(R + 'c_cands.parquet')
print(c.group_by('key', 'fit', (pl.col('namb') > 0).alias('amb')).agg(n=pl.len(), low=pl.col('lower').sum(), g=pl.col('p2g').median(), t=pl.col('p3').median()).sort('key', 'fit', 'amb'))
# France acceptance of these signatures among best candidates (v9y) — how big is the rejected residual
f = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fr_work.parquet', columns=['q', 's', 'key', 'sm', 'lower'])
v9 = pl.read_parquet(V + 'fr_v9y.parquet').with_columns(a=pl.lit(True))
keys = c['key'].unique().to_list()
x = f.filter(pl.col('key').is_in(keys) & ((pl.col('key').str.ends_with('NMISS')) | (pl.col('sm') == 'st'))).join(v9, on=['q', 's'], how='left').with_columns(pl.col('a').fill_null(False))
print(x.group_by('key').agg(n=pl.len(), acc=pl.col('a').mean(), rej=(~pl.col('a')).sum(), low_acc=pl.col('lower').filter(pl.col('a')).mean(), low_rej=pl.col('lower').filter(~pl.col('a')).mean()).sort('key'))
# ACRONYM: number-state ratio check in France vs US/India
fo = pl.read_parquet(f'{SCRATCH}/frfix2/census/fr_ops.parquet', columns=['q', 'nc', 'num'])
fo = fo.with_columns(acr=pl.col('nc').list.contains('ACRONYM'), dom=pl.col('nc').list.contains('DOMAIN'), dot=pl.col('nc').list.contains('LEGAL_DOT'), la=pl.col('nc').list.contains('LEGAL_ADD'),
                     numk=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'))
print('France: records by op and number state'); print(fo.group_by('numk').agg(acr=pl.col('acr').sum(), dom=pl.col('dom').sum(), dot=pl.col('dot').sum(), legal_add=pl.col('la').sum(), all=pl.len()).sort('numk'))
C = f'{SCRATCH}/frfix2/census/'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'num', 'nc']).with_columns(acr=pl.col('nc').list.contains('ACRONYM'), dom=pl.col('nc').list.contains('DOMAIN'), dot=pl.col('nc').list.contains('LEGAL_DOT'), la=pl.col('nc').list.contains('LEGAL_ADD'), numk=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'))
print('US/India: by grp and number state'); print(u.group_by('grp', 'numk').agg(acr=pl.col('acr').sum(), dom=pl.col('dom').sum(), dot=pl.col('dot').sum(), legal_add=pl.col('la').sum(), all=pl.len()).filter(pl.col('numk').is_in(['NSAME', 'UP', 'NMISS'])).sort('grp', 'numk'))
