import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../census"))
import polars as pl, collections
from ops import words, undot_legal, LEGAL
from common import WORK
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['business_name', 'country']).filter(pl.col('country') == 'France')['business_name'].to_list()
cnt = collections.Counter(w for n in s1 for w in set(words(undot_legal(n or ''))))
A = pl.read_parquet(f'{SCRATCH}/frfix2/fn/fn_add_all.parquet').filter(pl.col('tier') == 'A4')
def changed(qn, sn):
    q = [w for w in words(undot_legal(qn)) if w not in LEGAL]; s = set(w for w in words(undot_legal(sn)) if w not in LEGAL)
    return [w for w in q if w not in s]
ch = [changed(a, b) for a, b in zip(A['qn'].to_list(), A['sn'].to_list())]
realw = [any(cnt.get(w, 0) >= 50 for w in c) for c in ch]
A = A.with_columns(ch=pl.Series([' '.join(c) for c in ch]), real=pl.Series(realw), df=pl.Series([max([cnt.get(w, 0) for w in c] or [0]) for c in ch]))
print('A4 typo word is a frequent S1 word (df>=50):', A['real'].sum(), 'of', A.height, '| lowercase among them', A.filter(pl.col('real'))['lower'].sum())
print(A.filter(pl.col('real')).select('qn', 'sn', 'ch', 'df', 'p2g', 'p3').head(25))
