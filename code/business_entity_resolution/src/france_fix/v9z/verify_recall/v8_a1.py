import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(42); pl.Config.set_float_precision(3)
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
top = pl.read_parquet(f'{SCRATCH}/france/fr_top.parquet', columns=['q', 's', 'qn', 'qa', 'sn', 'sa', 'p2g', 'sn_2', 'p2_2', 'pat', 'rk'])
fa = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fn_add_all.parquet', columns=['q', 's', 'tier', 'key', 'lower', 'qn'])
# lowercase per tier before any filter (raw name test), vs the 'lower' column
fa = fa.with_columns(low2=pl.col('qn').str.contains(r'\p{L}') & (pl.col('qn') == pl.col('qn').str.to_lowercase()))
print(fa.group_by('tier').agg(n=pl.len(), low=pl.col('lower').sum(), low2=pl.col('low2').sum(), rate=pl.col('lower').mean()).sort('tier'))
print(fa.filter(pl.col('tier') == 'A1').group_by('key').agg(n=pl.len(), low=pl.col('lower').sum()))
x = add.join(top, on=['q', 's'], how='left')
for g in ['b_A1', 'b_A2', 'c_ACRONYM+LEGAL_DROP||NSAME', 'c_LEGAL_DOT||NSAME']:
    print(g); print(x.filter(pl.col('grp') == g).sample(14, seed=11).select('qn', 'sn', 'qa', 'sa', 'p2g', 'sn_2', 'p2_2'))
