# Independent artifact test: which address/format ops separate US/India TRUE from DECOY/WRONG records with a single word change at the
# same house number? Then apply those ops to France tiers (A1..A4, c groups) vs France references (veto set = decoys, accepted = true).
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
C = f'{SCRATCH}/frfix2/census/'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'))
x = u.filter((pl.col('num') == 'NSAME') & pl.col('bnc').is_in(['SWAP', 'ADD', 'TYPO', 'LEGAL_ADD', 'DROP']))
ex = x.explode('ops').filter(pl.col('ops').is_not_null())
nT = x.filter(pl.col('grp') == 'T').height; nF = x.filter(pl.col('grp') != 'T').height
t = ex.group_by('ops').agg(T=(pl.col('grp') == 'T').sum(), F=(pl.col('grp') != 'T').sum()).with_columns(
    rT=pl.col('T') / nT, rF=pl.col('F') / nF).with_columns(ratio=(pl.col('rT') + 1e-5) / (pl.col('rF') + 1e-5))
print('US/India NSAME single-change: T', nT, 'F', nF)
print(t.filter((pl.col('T') + pl.col('F')) >= 200).sort('ratio').filter((pl.col('ratio') < 0.5) | (pl.col('ratio') > 2)))
