# Word-level difference between vetoed record and each alternative S1; classify reassignment candidates.
import sys
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import words, undot_legal, LEGAL
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(330); pl.Config.set_fmt_str_lengths(40)
R = f'{SCRATCH}/frfix3/recall/'
c = pl.read_parquet(R + 'veto_alt.parquet')
v = pl.read_parquet(R + 'veto_status.parquet', columns=['q', 'sn', 'p2'])
c = c.join(v.rename({'p2': 'p2v'}), on='q')
def core(x):
    return [w for w in words(undot_legal(x or '')) if w not in LEGAL]
def wd(a, b):
    A, B = core(a), core(b)
    sa, sb = set(A), set(B)
    return len(sa - sb), len(sb - sa), (A[0] if A else '') == (B[0] if B else '')
res = [wd(a, b) for a, b in zip(c['qn'].to_list(), c['xn'].to_list())]
c = c.with_columns(w_add=pl.Series([r[0] for r in res]), w_drop=pl.Series([r[1] for r in res]), first_same=pl.Series([r[2] for r in res]))
c = c.with_columns(wdiff=pl.col('w_add') + pl.col('w_drop'))
c = c.with_columns(addr=pl.when(pl.col('same_st').fill_null(False) & pl.col('same_num').fill_null(False)).then(pl.lit('same'))
                   .when(pl.col('same_st').fill_null(False) & (pl.col('qnum').is_null() | (pl.col('qnum') == ''))).then(pl.lit('st_qnum_miss'))
                   .when(pl.col('same_st').fill_null(False)).then(pl.lit('st_num_diff'))
                   .when(pl.col('same_city')).then(pl.lit('city_only')).otherwise(pl.lit('other')))
c.write_parquet(R + 'veto_alt_wd.parquet')
print(c.group_by('addr', pl.col('wdiff').clip(0, 3)).agg(n=pl.len(), nq=pl.col('q').n_unique(), cand=pl.col('src_c').sum(), px=pl.col('px').mean()).sort('addr', 'wdiff'))
good = c.filter(pl.col('addr').is_in(['same', 'st_qnum_miss']) & (pl.col('wdiff') <= 2) & ((pl.col('wdiff') <= 1) | pl.col('first_same')))
print('plausible reassignments (address same or record number missing; <= 1 word swap/add):', good.height, 'records', good['q'].n_unique())
print(good.select('qn', 'sn', 'xn', 'qa', 'xa', 'px', 'p3x', 'gx', 'kx_now', 'wdiff').head(60))
