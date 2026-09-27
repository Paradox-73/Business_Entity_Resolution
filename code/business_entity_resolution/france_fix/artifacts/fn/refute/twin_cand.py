import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
import polars as pl
from common import WORK, id_to_int
from fr_restore import street
print(street('25 RUE de Valmy, Lille, Hauts-de-France'), street('RUE DE VALMY, LILLE'), street('Impasse Des Tardones, Pornic, Pays de la Loire'))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(3)
exec(open('twin_nmiss.py').read().split("per = sib.group_by")[0].split("pl.Config.set_tbl_rows")[0])
