import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
OUT = 'C:/ber_scratch/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_work.parquet', columns=['q', 's', 'qn', 'sn', 'p2g', 'p3', 'p9', 'acc9', 'key', 'num2', 'sm', 'stw', 'p2_2'])
d = d.join(pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'p2', 'acc', 's9acc', 'q_acc_else']), on=['q', 's'])
# own lowercase: record has letters, record == lower(record), S1 != lower(S1)
d = d.with_columns(low=(pl.col('qn') == pl.col('qn').str.to_lowercase()) & (pl.col('sn') != pl.col('sn').str.to_lowercase()) & pl.col('qn').str.contains(r'[^\W\d_]'))
veto = pl.read_parquet('C:/ber_scratch/frfix/namechg/veto_set.parquet', columns=['q', 's']).with_columns(veto=pl.lit(True))
addv2 = pl.read_parquet('C:/ber_scratch/frfix/build/add_set_v2_final.parquet', columns=['q', 's']).with_columns(gnadd=pl.lit(True))
d = d.join(veto, on=['q', 's'], how='left').join(addv2, on=['q', 's'], how='left').with_columns(pl.col('veto').fill_null(False), pl.col('gnadd').fill_null(False))
A1k = ['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME']
x = d.filter(pl.col('key').is_in(A1k) & (pl.col('sm') == 'st'))
bh = (pl.col('p2g') >= 0.5) & (pl.col('p3') >= 0.5)
x = x.with_columns(state=pl.when(pl.col('acc9')).then(pl.lit('acc9')).when(pl.col('q_acc_else')).then(pl.lit('rej:q matched elsewhere'))
                   .when(pl.col('veto')).then(pl.lit('rej:vetoed')).when(bh).then(pl.lit('rej:both_hi other')).otherwise(pl.lit('rej:A1-like')))
print(x.group_by('state', 'key').agg(n=pl.len(), low=pl.col('low').mean(), nlow=pl.col('low').sum(), g=pl.col('p2g').median(), t=pl.col('p3').median(),
      v7acc=pl.col('acc').mean(), gnadd=pl.col('gnadd').mean()).sort('key', 'state'))
x.write_parquet('fr_a1sig.parquet')
d.select('q', 's', 'low', 'veto', 'gnadd', 'acc', 'q_acc_else').write_parquet('fr_low_own.parquet')
