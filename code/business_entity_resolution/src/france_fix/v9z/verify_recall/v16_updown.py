# UP vs DOWN house-number moves per single-change key: decoys move the number UP only, true records UP ~= DOWN.
# An UP excess in a key means the France decoy generator uses that change; then the same key at the same number can hide decoys.
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
f = pl.read_parquet(C + 'fr_ops.parquet', columns=['q', 'ops', 'nc', 'num'])
def prep(d):
    return d.with_columns(lower=pl.col('ops').list.contains('n_lower'), numk=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
                          bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                          wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' ')).with_columns(
                          k2=pl.col('bnc') + '|' + pl.col('wcls'))
f = prep(f)
keys = ['|', 'LEGAL_DOT|', 'ACRONYM|', 'ACRONYM+LEGAL_DROP|', 'DOMAIN|', 'LEGAL_ADD|', 'LEGAL_DROP|', 'TYPO|', 'SWAP|n_swap:noise', 'SWAP|n_swap:desc>noise', 'ADD|n_add:noise', 'SWAP|n_swap:desc', 'SQUASH|']
t = f.filter(pl.col('k2').is_in(keys)).group_by('k2', 'numk').agg(n=pl.len(), low=pl.col('lower').mean()).pivot(on='numk', index='k2', values='n').fill_null(0)
t = t.with_columns(up_down=pl.col('UP') / pl.col('DOWN'), up_same=pl.col('UP') / pl.col('NSAME'))
print('FRANCE (best candidates)'); print(t.select('k2', 'NSAME', 'UP', 'DOWN', 'NMISS', 'up_down', 'up_same').sort('k2'))
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = prep(u)
tu = u.filter(pl.col('k2').is_in(keys)).group_by('k2', 'grp', 'numk').agg(n=pl.len()).pivot(on='numk', index=['k2', 'grp'], values='n').fill_null(0)
tu = tu.with_columns(up_down=pl.col('UP') / pl.col('DOWN'), up_same=pl.col('UP') / pl.col('NSAME'))
print('US/INDIA by truth group'); print(tu.select('k2', 'grp', 'NSAME', 'UP', 'DOWN', 'NMISS', 'up_down', 'up_same').sort('k2', 'grp'))
