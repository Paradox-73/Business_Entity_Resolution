# cleaned add set: drop A1 pairs whose record swaps IN a non-noise word (noise>other / multi-swap), A1 decoy-vocabulary words, A4 Ets->Fetes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
from common import WORK  # noqa: E402
import polars as pl
from common import WORK
sub = pl.read_parquet('fn_sub.parquet')
keep = sub.filter(pl.col('tier').str.starts_with('A') & ~pl.col('sub').is_in(['A1q', 'A1x', 'A4q'])).select('q', 's')
print('clean add set', keep.height)
keep.write_parquet(f'{WORK}/frfix2/fn_add_set_clean.parquet')
sc = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet')
n0 = sc.height
sc = sc.join(keep.with_columns(a=pl.lit(True)), on=['q', 's'], how='left').with_columns(
    p2=pl.when(pl.col('a').fill_null(False)).then(pl.lit(0.95)).otherwise(pl.col('p2')).cast(pl.Float32)).drop('a')
assert sc.height == n0
sc.write_parquet(f'{WORK}/frfix2/test_scores_v9b_fnadd_clean.parquet')
print('written', sc.height)
