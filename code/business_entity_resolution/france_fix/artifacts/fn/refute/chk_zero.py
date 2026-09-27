import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from common import WORK
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
add = pl.read_parquet('E:/Projects/Amazon ML Challenge/work/frfix2/fn_add_set.parquet')
old = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet', columns=['q', 's', 'p2']).join(add, on=['q', 's'])
A = pl.read_parquet('C:/ber_scratch/frfix2/fn/fn_add_all.parquet', columns=['q', 's', 'tier', 'qn', 'sn', 'p2g', 'p3'])
veto = pl.read_parquet('C:/ber_scratch/frfix/namechg/veto_set.parquet', columns=['q', 's']).with_columns(veto=pl.lit(True))
x = old.join(A, on=['q', 's']).join(veto, on=['q', 's'], how='left').with_columns(pl.col('veto').fill_null(False))
print(x.group_by('tier').agg(n=pl.len(), p2zero=(pl.col('p2') == 0).sum(), vetoed=pl.col('veto').sum(), p2med=pl.col('p2').median()).sort('tier'))
print(x.filter(pl.col('veto')).head(10).select('tier', 'qn', 'sn', 'p2', 'p2g', 'p3'))
print(x.filter((pl.col('p2') == 0) & ~pl.col('veto')).head(10).select('tier', 'qn', 'sn', 'p2', 'p2g', 'p3'))
