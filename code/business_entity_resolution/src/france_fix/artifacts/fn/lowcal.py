# Lowercase calibration: per (name signature without LOWER, word-op classes, number state) the lowercase share of
# US/India true (T) and false (D+W) pairs, and the implied true share of France accepted / rejected pairs.
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(150); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
OUT = f'{SCRATCH}/frfix2/fn/'
def prep(df):
    return df.with_columns(
        lower=pl.col('ops').list.contains('n_lower'),
        num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
        bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
        wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(
    pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = prep(u).with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'), T=pl.col('grp') == 'T')
cal = u.group_by('key').agg(nT=pl.col('T').sum(), nF=(~pl.col('T')).sum(), trU=pl.col('T').filter(pl.col('country') == 'US').mean(),
                            trI=pl.col('T').filter(pl.col('country') == 'India').mean(),
                            lowT=pl.col('lower').filter(pl.col('T')).mean(), lowF=pl.col('lower').filter(~pl.col('T')).mean())
cal.write_parquet(OUT + 'lowcal_usi.parquet')
f = pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'acc9', 'nc', 'num', 'ops', 'p2g', 'p3']).join(
    pl.read_parquet(OUT + 'fr_street.parquet', columns=['q', 's', 'sm', 'sm2']), on=['q', 's'])
f = prep(f).with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'))
f.select('q', 's', 'key', 'bnc', 'wcls', 'num2', 'lower', 'sm').write_parquet(OUT + 'fr_key2.parquet')
# France: NSAME only when street also matches
f = f.filter((pl.col('num2') != 'NSAME') | (pl.col('sm') == 'st'))
g = f.group_by('key').agg(n=pl.len(), rej=(~pl.col('acc9')).sum(), acc=pl.col('acc9').mean(),
                          la=pl.col('lower').filter(pl.col('acc9')).mean(), lr=pl.col('lower').filter(~pl.col('acc9')).mean(),
                          g_r=pl.col('p2g').filter(~pl.col('acc9')).median(), t_r=pl.col('p3').filter(~pl.col('acc9')).median())
g = g.join(cal, on='key', how='left').with_columns(
    fT_rej=((pl.col('lr') - pl.col('lowF')) / (pl.col('lowT') - pl.col('lowF'))),
    fT_acc=((pl.col('la') - pl.col('lowF')) / (pl.col('lowT') - pl.col('lowF'))))
g.write_csv(OUT + 'fr_lowcal.csv')
print(g.filter((pl.col('rej') >= 150) & ~pl.col('key').str.contains(r'\|(UP|DOWN)$') & (pl.col('trU').fill_null(0) + pl.col('trI').fill_null(0) > 1.3))
      .sort('rej', descending=True).drop('n'))
