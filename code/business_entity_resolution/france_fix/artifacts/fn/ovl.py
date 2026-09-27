# word overlap between record name and S1 name (legal forms and stopwords removed), France and US/India
import sys
sys.path.insert(0, 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census')
import polars as pl
from ops import words, undot_legal, LEGAL, STOPW, _match
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
def ov(qn, sn):
    qw = [w for w in words(undot_legal(qn or '')) if w not in LEGAL and w not in STOPW]
    sw = [w for w in words(undot_legal(sn or '')) if w not in LEGAL and w not in STOPW]
    if not qw or not sw:
        return -1, len(qw), len(sw)
    m = sum(1 for a in sw if any(_match(a, b) for b in qw))
    return m, len(qw), len(sw)
for tag, path, cols in [('fr', OUT + 'fr_base.parquet', ['q', 's', 'qn', 'sn']),
                        ('usi', 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france/usi_top.parquet', ['q', 's', 'qn', 'sn'])]:
    d = pl.read_parquet(path, columns=cols)
    r = [ov(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())]
    d.select('q', 's').with_columns(ovm=pl.Series([x[0] for x in r], dtype=pl.Int16), nqw=pl.Series([x[1] for x in r], dtype=pl.Int16),
                                    nsw=pl.Series([x[2] for x in r], dtype=pl.Int16)).write_parquet(OUT + f'ovl_{tag}.parquet')
    print(tag, 'done')
