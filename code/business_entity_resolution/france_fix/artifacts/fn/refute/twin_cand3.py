import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from common import WORK
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(3)
tp = pl.read_parquet('fn_twin_pairs.parquet')
sc = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet', columns=['q', 's', 'p2']).filter(pl.col('q').is_in(tp['q'].implode()))
tp = tp.join(sc.rename({'s': 's2', 'p2': 'p2_sib'}), on=['q', 's2'], how='left').join(sc.rename({'p2': 'p2_best'}), on=['q', 's'], how='left')
tp = tp.with_columns(in_cand=pl.col('p2_sib').is_not_null())
print(tp.group_by('tier').agg(n=pl.len(), q=pl.col('q').n_unique(), in_cand=pl.col('in_cand').mean(), sib_ge_half=(pl.col('p2_sib') >= 0.5 * pl.col('p2_best')).mean(), p2_sib=pl.col('p2_sib').median(), p2_best=pl.col('p2_best').median()).sort('tier'))
print(tp.filter(pl.col('in_cand') & pl.col('tier').is_in(['A2', 'A3'])).sample(10, seed=2).select('tier', 'qn', 'sn', 'n2', 'p2_best', 'p2_sib'))
print(tp.filter(~pl.col('in_cand') & pl.col('tier').is_in(['A2', 'A3'])).sample(10, seed=2).select('tier', 'qn', 'sn', 'n2', 'a2', 'p2_best'))
