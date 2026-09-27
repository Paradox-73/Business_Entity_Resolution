import polars as pl, sys
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
from ops import words, undot_legal, LEGAL, _match
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(330); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
OUT = 'C:/ber_scratch/frfix2/fn/'
def nw_(qn, sn):
    qx = [w for w in words(undot_legal(qn or '')) if w not in LEGAL]; sx = [w for w in words(undot_legal(sn or '')) if w not in LEGAL]
    used = [False] * len(qx)
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None: j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
        if j is not None: used[j] = True
    return ' '.join(sorted({qx[j] for j in range(len(qx)) if not used[j]}))
d = pl.read_parquet(OUT + 'fr_work.parquet', columns=['q', 's', 'qn', 'sn', 'acc9', 'num2', 'key', 'sm', 'p2g', 'p3'])
d = d.join(pl.read_parquet('fr_low_own.parquet', columns=['q', 's', 'low', 'veto']), on=['q', 's'])
d = d.join(pl.read_parquet('C:/ber_scratch/frfix2/census/fr_ops.parquet', columns=['q', 's', 'ops']), on=['q', 's'], how='left')
d = d.filter(pl.col('key').str.contains(r'^(ADD|SWAP|ADD\+SWAP)\|'))
d = d.with_columns(nw=pl.Series([nw_(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())], dtype=pl.Utf8))
S = ['participations', 'holding', 'international', 'distribution', 'groupe', 'developpement', 'france']
d = d.filter(pl.col('nw').is_in(S))
d = d.with_columns(op=pl.col('key').str.split('|').list.first(),
                   ast=pl.when(pl.col('num2').is_in(['UP', 'DOWN'])).then(pl.lit('numchg')).when((pl.col('num2') == 'NSAME') & (pl.col('sm') == 'st')).then(pl.lit('same')).when(pl.col('num2') == 'NMISS').then(pl.lit('nmiss')).otherwise(pl.lit('other')),
                   last=pl.col('qn').str.to_lowercase().str.strip_chars().str.contains(r'(d[ée]veloppement|groupe|france|holding|participations|international|distribution)$'))
ag = [pl.len().alias('n'), pl.col('acc9').mean().alias('acc'), pl.col('low').mean().alias('low'), pl.col('last').mean().alias('at_end'),
      pl.col('ops').list.contains('n_hyphen').mean().alias('hy'), pl.col('ops').list.contains('n_dblspace').mean().alias('dbl'), pl.col('ops').list.contains('n_upper').mean().alias('up'), pl.col('ops').list.contains('n_bracket').mean().alias('br')]
print(d.group_by('nw', 'ast', 'op').agg(ag).filter(pl.col('n') >= 100).sort('nw', 'ast', 'op'))
print(d.filter((pl.col('ast') == 'numchg') & (pl.col('nw') == 'developpement')).sample(10, seed=3).select('qn', 'sn', 'key'))
print(d.filter((pl.col('ast') == 'same') & (pl.col('nw') == 'holding')).head(10).select('qn', 'sn', 'key', 'acc9', 'p2g', 'p3'))
d.drop('ops').write_parquet('structw.parquet')
