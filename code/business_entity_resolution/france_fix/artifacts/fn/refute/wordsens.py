# per-new-word lowercase sensitivity: decoy lowercase rate among number-changed rejected France records that add the same word,
# then expected lowercase count of each tier if it were all decoys vs observed
import polars as pl, sys
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
from ops import words, undot_legal, LEGAL, _match
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
OUT = 'C:/ber_scratch/frfix2/fn/'
def newwords(qn, sn):
    qx = [w for w in words(undot_legal(qn or '')) if w not in LEGAL]; sx = [w for w in words(undot_legal(sn or '')) if w not in LEGAL]
    used = [False] * len(qx)
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None: j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
        if j is not None: used[j] = True
    nw = sorted({qx[j] for j in range(len(qx)) if not used[j]})
    return ' '.join(nw)
d = pl.read_parquet(OUT + 'fr_work.parquet', columns=['q', 's', 'qn', 'sn', 'acc9', 'num2', 'key', 'sm'])
d = d.join(pl.read_parquet('fr_low_own.parquet', columns=['q', 's', 'low']), on=['q', 's'])
ref = d.filter(pl.col('num2').is_in(['UP', 'DOWN']) & ~pl.col('acc9') & pl.col('key').str.contains(r'^(ADD|SWAP|ADD\+SWAP)\|'))
ref = ref.with_columns(nw=pl.Series([newwords(a, b) for a, b in zip(ref['qn'].to_list(), ref['sn'].to_list())], dtype=pl.Utf8))
rw = ref.group_by('nw').agg(nref=pl.len(), rlow=pl.col('low').mean()).filter(pl.col('nref') >= 300).sort('nref', descending=True)
print('decoy lowercase rate by new word (number-changed, rejected, word add/swap):'); print(rw.head(30))
print('overall ref', ref['low'].mean(), ref.height)
A = pl.read_parquet(OUT + 'fn_add_all.parquet', columns=['q', 's', 'tier', 'qn', 'sn']).join(pl.read_parquet('fr_low_own.parquet', columns=['q', 's', 'low']), on=['q', 's'])
A = A.with_columns(nw=pl.Series([newwords(a, b) for a, b in zip(A['qn'].to_list(), A['sn'].to_list())], dtype=pl.Utf8))
A = A.join(rw.select('nw', 'rlow', 'nref'), on='nw', how='left')
print(A.group_by('tier').agg(n=pl.len(), nlow=pl.col('low').sum(), cover=pl.col('rlow').is_not_null().mean(), exp_if_decoy_cov=pl.col('rlow').sum(),
      nlow_cov=pl.col('low').filter(pl.col('rlow').is_not_null()).sum(), n_cov=pl.col('rlow').is_not_null().sum()).sort('tier'))
print(A.filter(pl.col('tier').is_in(['A1', 'A2'])).group_by('tier', 'nw').agg(n=pl.len(), nlow=pl.col('low').sum(), rlow=pl.col('rlow').first(), nref=pl.col('nref').first()).sort('n', descending=True).head(30))
A.select('q', 's', 'tier', 'nw', 'rlow', 'low').write_parquet('fn_nw.parquet'); rw.write_parquet('rw.parquet')
