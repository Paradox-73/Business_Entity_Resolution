# Is the lowercase test blind for records whose added/swapped word is 'Developpement' (or other noise words)?
import polars as pl, re, sys
sys.path.insert(0, '.')
from partlow import case_kind
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_work.parquet', columns=['q', 's', 'qn', 'sn', 'p2g', 'p3', 'acc9', 'key', 'num2', 'sm'])
d = d.with_columns(ql=pl.col('qn').str.to_lowercase(), sl=pl.col('sn').str.to_lowercase())
W = {'dev': r'd[ée]veloppement', 'groupe': r'\bgroupe\b', 'france': r'\bfrance\b', 'services': r'\bservices?\b', 'fils': r'\bfils\b', 'cie': r'\bcie\b', 'associes': r'associ[ée]s', 'and': r'&'}
rows = []
for w, rx in W.items():
    x = d.filter(pl.col('ql').str.contains(rx) & ~pl.col('sl').str.contains(rx))
    x = x.with_columns(ck=pl.Series([case_kind(a, b) for a, b in zip(x['qn'].to_list(), x['sn'].to_list())], dtype=pl.Utf8))
    x = x.with_columns(nst=pl.when(pl.col('num2').is_in(['UP', 'DOWN'])).then(pl.lit('numchg')).when((pl.col('num2') == 'NSAME') & (pl.col('sm') == 'st')).then(pl.lit('same addr')).otherwise(pl.lit('other')))
    g = x.group_by('nst', 'acc9').agg(n=pl.len(), full=(pl.col('ck') == 'full').mean(), part=(pl.col('ck') == 'part').mean(), nfull=(pl.col('ck') == 'full').sum(), npart=(pl.col('ck') == 'part').sum())
    rows.append(g.with_columns(word=pl.lit(w)))
print(pl.concat(rows).sort('word', 'nst', 'acc9'))
# reference: descriptor added (decoy word) in number-changed records
x = d.filter(pl.col('ql').str.contains(r'\b(club|ecole|école|comit[ée]|amicale|centre)\b') & ~pl.col('sl').str.contains(r'\b(club|ecole|école|comit[ée]|amicale|centre)\b'))
x = x.with_columns(ck=pl.Series([case_kind(a, b) for a, b in zip(x['qn'].to_list(), x['sn'].to_list())], dtype=pl.Utf8))
print('descriptor-in records:', x.group_by(pl.col('num2').is_in(['UP', 'DOWN']).alias('numchg'), 'acc9').agg(n=pl.len(), full=(pl.col('ck') == 'full').mean(), part=(pl.col('ck') == 'part').mean()).sort('numchg', 'acc9'))
