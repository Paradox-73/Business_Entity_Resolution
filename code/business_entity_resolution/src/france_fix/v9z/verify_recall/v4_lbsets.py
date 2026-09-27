# France pair sets of past uploads, their differences vs v7ens, overlap with the recall add set, and lowercase rates
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(4)
V = f'{SCRATCH}/frfix3/verify_recall/'
SUB = f'{ROOT}/submissions/'
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'business_name', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'), sn='business_name')
def load(v):
    d = pl.read_csv(SUB + v + '/matching_results.tsv', separator='\t', schema_overrides={'source1_entity_id': pl.Utf8, 'matched_entity_ids': pl.Utf8})
    d = d.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
    d = d.select(s=id_to_int('source1_entity_id'), q=pl.col('matched_entity_ids').str.split(',')).explode('q').with_columns(q=id_to_int('q'))
    return d.join(s1.select('s'), on='s')
out = {}
for v in ['v7ens', 'v7m', 'v7i', 'v7j', 'v9b', 'v9e', 'v9f', 'v9y']:
    out[v] = load(v); print(v, out[v].height)
    out[v].write_parquet(V + f'fr_{v}.parquet')
