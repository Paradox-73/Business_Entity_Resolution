# stricter sibling ambiguity for number-absent tiers: edit distance counts legal-form changes; sibling must explain the record at least as well
import sys
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../census"))
import polars as pl
from ops import words, undot_legal, LEGAL, _match
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(3)
tp = pl.read_parquet('fn_twin_pairs.parquet')
CAN = {'societe': 'societe'}
def parts(n):
    w = words(undot_legal(n or ''))
    return [x for x in w if x not in LEGAL], sorted(set(x for x in w if x in LEGAL))
def um(a, b):
    b = list(b); u = 0
    for w in a:
        j = next((j for j, v in enumerate(b) if w == v or (len(w) > 3 and _match(w, v))), None)
        if j is None: u += 1
        else: b.pop(j)
    return u
def dist(qn, n):
    cq, lq = parts(qn); c, l = parts(n)
    d = max(um(c, cq), um(cq, c))            # a swap counts once
    if l != lq: d += 1                        # legal form dropped / added / changed
    return d
tp = tp.with_columns(d1=pl.Series([dist(a, b) for a, b in zip(tp['qn'].to_list(), tp['sn'].to_list())]),
                     d2=pl.Series([dist(a, b) for a, b in zip(tp['qn'].to_list(), tp['n2'].to_list())]))
tp = tp.with_columns(amb=pl.col('d2') <= pl.col('d1'))
per = tp.group_by('q', 's', 'tier').agg(namb=pl.col('amb').sum())
A = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fn_add_all.parquet', columns=['q', 's', 'tier'])
x = A.join(per.drop('tier'), on=['q', 's'], how='left').with_columns(pl.col('namb').fill_null(0))
print(x.group_by('tier').agg(n=pl.len(), amb=(pl.col('namb') > 0).sum(), exp_lost=(pl.col('namb') / (pl.col('namb') + 1)).sum()).with_columns(frac_lost=pl.col('exp_lost') / pl.col('n')).sort('tier'))
print(tp.filter(pl.col('amb') & pl.col('tier').is_in(['A2', 'A3'])).sample(12, seed=3).select('tier', 'qn', 'sn', 'n2', 'd1', 'd2'))
x.write_parquet('fn_amb.parquet')
