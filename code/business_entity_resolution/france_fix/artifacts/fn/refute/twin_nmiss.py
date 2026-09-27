# number-absent tiers: other S1 rows on the same street (any number) whose name explains the record at least as well
import sys
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
import polars as pl
from common import WORK, id_to_int
from fr_restore import street
from ops import words, undot_legal, LEGAL, _match
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(3)
OUT = 'C:/ber_scratch/frfix2/fn/'
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'business_name', 'business_address', 'country']).filter(pl.col('country') == 'France')
def skey(a):
    n, st = street(a)
    comps = [c.strip().lower() for c in (a or '').split(',')]
    city = comps[1] if len(comps) > 1 else ''
    return f'{st}|{city}' if st else None
s1 = s1.select(s=id_to_int('entity_id'), name='business_name', addr='business_address')
s1 = s1.with_columns(sk=pl.Series([skey(a) for a in s1['addr'].to_list()], dtype=pl.Utf8))
A = pl.read_parquet(OUT + 'fn_add_all.parquet', columns=['q', 's', 'tier', 'qn', 'sn', 'key'])
x = A.join(s1.select('s', 'sk'), on='s', how='left')
sib = x.filter(pl.col('sk').is_not_null()).join(s1.select(s2='s', sk='sk', n2='name', a2='addr'), on='sk').filter(pl.col('s2') != pl.col('s'))
def core(n): return [w for w in words(undot_legal(n or '')) if w not in LEGAL]
def legal(n): return sorted(w for w in words(undot_legal(n or '')) if w in LEGAL)
def um(a, b):
    b = list(b); u = 0
    for w in a:
        j = next((j for j, v in enumerate(b) if w == v or (len(w) > 3 and _match(w, v))), None)
        if j is None: u += 1
        else: b.pop(j)
    return u
def plaus(qn, n2, sn):
    q, c2, c1 = core(qn), core(n2), core(sn)
    d2 = um(c2, q) + um(q, c2); d1 = um(c1, q) + um(q, c1)
    lq, l1, l2 = legal(qn), legal(sn), legal(n2)
    # legal form evidence: if the record carries a legal form, it must agree
    pen2 = 1 if (lq and l2 and lq != l2) else 0; pen1 = 1 if (lq and l1 and lq != l1) else 0
    return d2 + pen2 <= d1 + pen1
sib = sib.with_columns(pl_=pl.Series([plaus(a, b, c) for a, b, c in zip(sib['qn'].to_list(), sib['n2'].to_list(), sib['sn'].to_list())], dtype=pl.Boolean))
per = sib.group_by('q', 's').agg(nsib=pl.len(), npl=pl.col('pl_').sum())
x = x.join(per, on=['q', 's'], how='left').with_columns(pl.col('nsib').fill_null(0), pl.col('npl').fill_null(0))
print(x.group_by('tier', 'key').agg(n=pl.len(), street_sib=(pl.col('nsib') > 0).mean(), plaus=(pl.col('npl') > 0).mean(), exp_lost=(pl.col('npl') / (pl.col('npl') + 1)).sum())
      .filter(pl.col('n') >= 20).sort('tier', 'n', descending=[False, True]))
print(x.group_by('tier').agg(n=pl.len(), plaus=(pl.col('npl') > 0).mean(), exp_lost=(pl.col('npl') / (pl.col('npl') + 1)).sum()).sort('tier'))
print(sib.filter(pl.col('pl_') & pl.col('tier').is_in(['A2', 'A3'])).sample(12, seed=1).select('tier', 'qn', 'sn', 'n2', 'a2'))
x.write_parquet('fn_twin_street.parquet'); sib.filter(pl.col('pl_')).select('q', 's', 's2', 'tier', 'qn', 'sn', 'n2', 'a2').write_parquet('fn_twin_pairs.parquet')
