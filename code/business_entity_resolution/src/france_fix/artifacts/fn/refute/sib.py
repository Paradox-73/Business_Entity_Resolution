# sibling S1 rows at the same address key that could equally be the record's parent (one-word swap/add away from the record)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../census"))
import polars as pl
from common import WORK, id_to_int
from ops import words, undot_legal, LEGAL, _match
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(3)
OUT = f'{SCRATCH}/frfix2/fn/'
ak = pl.read_parquet(OUT + 'akey_test.parquet')
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'business_name', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'), name='business_name')
ak = ak.join(s1, on='s')
A = pl.read_parquet(OUT + 'fn_add_all.parquet', columns=['q', 's', 'tier', 'qn', 'sn'])
x = A.join(ak.select('s', 'akey', 'nakey'), on='s', how='left')
sib = x.filter(pl.col('nakey') >= 2).join(ak.select(s2='s', akey='akey', n2='name'), on='akey').filter(pl.col('s2') != pl.col('s'))
def core(n): return [w for w in words(undot_legal(n or '')) if w not in LEGAL]
def unmatched(a, b):
    """number of words of a not matched in b"""
    b = list(b); u = 0
    for w in a:
        j = next((j for j, v in enumerate(b) if _match(w, v)), None)
        if j is None: u += 1
        else: b.pop(j)
    return u
def plausible(qn, n2, sn):
    q, c2, c1 = core(qn), core(n2), core(sn)
    # sibling explains the record at least as well as the chosen S1: unmatched S1 words + unmatched record words
    d2 = unmatched(c2, q) + unmatched(q, c2); d1 = unmatched(c1, q) + unmatched(q, c1)
    return d2 <= d1
sib = sib.with_columns(pl_=pl.Series([plausible(a, b, c) for a, b, c in zip(sib['qn'].to_list(), sib['n2'].to_list(), sib['sn'].to_list())]))
per = sib.group_by('q', 's').agg(nsib=pl.len(), nplaus=pl.col('pl_').sum())
x = x.join(per, on=['q', 's'], how='left').with_columns(pl.col('nsib').fill_null(0), pl.col('nplaus').fill_null(0))
print(x.group_by('tier').agg(n=pl.len(), has_sib=(pl.col('nsib') > 0).mean(), has_plaus=(pl.col('nplaus') > 0).mean(), n_plaus=(pl.col('nplaus') > 0).sum(),
      exp_share_lost=(pl.col('nplaus') / (pl.col('nplaus') + 1)).sum()).sort('tier'))
print(sib.filter(pl.col('pl_')).head(15).select('tier', 'qn', 'sn', 'n2'))
x.write_parquet('fn_sib.parquet')
