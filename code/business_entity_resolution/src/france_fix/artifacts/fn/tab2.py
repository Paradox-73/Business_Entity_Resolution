import polars as pl, sys
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40); pl.Config.set_float_precision(3)
OUT = f'{SCRATCH}/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'p9', 'p2g', 'p3', 'acc9', 'nc', 'num', 'ops', 'p2_2'])
d = d.join(pl.read_parquet(OUT + 'fr_street.parquet'), on=['q', 's'])
d = d.with_columns(num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'), lower=pl.col('ops').list.contains('n_lower'),
                   bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'))
d = d.with_columns(key=pl.col('bnc') + '|' + pl.col('num2') + '|' + pl.col('sm'))
d.select('q', 's', 'key', 'bnc', 'num2', 'sm', 'lower').write_parquet(OUT + 'fr_keys.parquet')
g = d.group_by('key').agg(n=pl.len(), rej=(~pl.col('acc9')).sum(), acc=pl.col('acc9').mean(),
        low_a=pl.col('lower').filter(pl.col('acc9')).mean(), low_r=pl.col('lower').filter(~pl.col('acc9')).mean(),
        p2g_r=pl.col('p2g').filter(~pl.col('acc9')).median(), p3_r=pl.col('p3').filter(~pl.col('acc9')).median(),
        tw_r=(pl.col('p2_2') > 0.3).filter(~pl.col('acc9')).mean(), st2_r=pl.col('sm2').is_in(['st']).filter(~pl.col('acc9')).mean())
g = g.sort('rej', descending=True)
g.write_csv(OUT + 'fr_rej_keys.csv')
print(g.filter(~pl.col('key').str.contains(r'\|(UP|DOWN)\|')).filter(pl.col('rej') >= 200))
