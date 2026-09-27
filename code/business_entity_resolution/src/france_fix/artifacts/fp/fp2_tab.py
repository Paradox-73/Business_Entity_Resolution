"""Build per-pair tables: US/India reference (sig_all) and France (fr_ops + v9b acceptance), base signature = name
changes without LOWER + house-number class; low = record name all-lowercase."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
C = f'{SCRATCH}/frfix2/census/'
F = f'{SCRATCH}/frfix2/fp/'


def numc(col):
    return (pl.when(pl.col(col).str.starts_with('UP') & (pl.col(col) != 'UPbig')).then(pl.lit('UP'))
              .when(pl.col(col).str.starts_with('DOWN') & (pl.col(col) != 'DOWNbig')).then(pl.lit('DOWN'))
              .otherwise(pl.col(col)))


def add_base(d):
    return d.with_columns(low=pl.col('nc').list.contains('LOWER'),
                          base=pl.col('nc').list.filter(pl.element() != 'LOWER').list.sort().list.join('+'),
                          nchg=pl.col('nc').list.filter(pl.element() != 'LOWER').list.len()).with_columns(
        bsig=pl.col('base') + '|' + pl.col('numc'))


u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'nc', 'num', 'numc'])
u = add_base(u)
u.write_parquet(F + 'usi_sig.parquet')

fr = pl.read_parquet(C + 'fr_ops.parquet')
acc = pl.read_parquet(F + 'acc_v9b.parquet', columns=['q', 's', 'p2']).rename({'p2': 'p2v9b'})
veto = pl.read_parquet(f'{SCRATCH}/frfix/namechg/veto_set.parquet', columns=['q', 's'])
add = pl.read_parquet(f'{SCRATCH}/frfix/build/add_set_v2_final.parquet', columns=['q', 's'])
fr = (fr.join(acc.with_columns(acc9=pl.lit(True)), on=['q', 's'], how='left')
        .join(veto.with_columns(veto=pl.lit(True)), on=['q', 's'], how='left')
        .join(add.with_columns(gadd=pl.lit(True)), on=['q', 's'], how='left')
        .with_columns(pl.col('acc9').fill_null(False), pl.col('veto').fill_null(False), pl.col('gadd').fill_null(False),
                      numc=numc('num')))
fr = add_base(fr)
fr.write_parquet(F + 'fr_sig.parquet')
print(fr.select(n=pl.len(), acc9=pl.col('acc9').sum(), acc7=pl.col('acc').sum(), veto=pl.col('veto').sum(),
                veto_acc7=(pl.col('veto') & pl.col('acc')).sum(), gadd=pl.col('gadd').sum()))
