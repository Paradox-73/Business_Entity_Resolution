import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from common import WORK
from fr_restore import street
print(street('25 RUE de Valmy, Lille, Hauts-de-France'), '|', street('RUE DE VALMY, LILLE'), '|', street('7 Impasse des Tardones, Pornic'))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(3)
x = pl.read_parquet('fn_twin_street.parquet')
# rebuild plausible sibling pairs from twin_nmiss (saved only aggregates) -> recompute quickly for A2/A3 only
