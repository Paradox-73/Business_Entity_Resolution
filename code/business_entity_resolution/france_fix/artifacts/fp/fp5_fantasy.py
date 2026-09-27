"""Records whose name shares no word with the S1 name ('fantasy' / unrelated names): US/India label rates by number class,
France acceptance, lowercase share and number class."""
import sys, re
import polars as pl
sys.path.insert(0, 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census')
from ops import words, undot_legal, LEGAL, STOPW
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
TMP = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/'
LINK = re.compile(r'(?i)\b(dba|d/b/a|d\.b\.a|doing business as|trading as|t/a|aka|a/k/a|fka|f/k/a|formerly)\b')


def kind(qn, sn):
    if qn is None or not qn.strip():
        return 'empty'
    if qn.strip()[:1] in '#@':
        return 'handle'
    if LINK.search(qn):
        return 'linked'
    qw = [w for w in words(undot_legal(qn)) if w not in LEGAL and w not in STOPW]
    sw = [w for w in words(undot_legal(sn or '')) if w not in LEGAL and w not in STOPW]
    if not qw or not sw:
        return 'legal_only'
    qs, ss = set(qw), set(sw)
    if qs & ss:
        return 'overlap'
    # partial overlap: a record word containing / contained in an S1 word (squash, acronym, typos)
    j = ''.join(sw)
    for w in qw:
        if len(w) >= 3 and (w in j or any(len(x) >= 3 and (x in w or w[:4] == x[:4]) for x in sw)):
            return 'partial'
    if all(len(w) <= 4 for w in qw) and len(qw) <= 2:
        return 'short'           # acronym-like
    return 'fantasy'


def tag(d):
    return d.with_columns(fk=pl.Series([kind(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())]))


u = pl.read_parquet(TMP + 'france/usi_top.parquet', columns=['q', 's', 'label', 'p', 'qn', 'sn', 'country'])
u = tag(u)
us = pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'numc', 'low'])
u = u.join(us, on='q', how='left')
u.select('q', 's', 'fk', 'grp', 'numc', 'low', 'p', 'country').write_parquet(F + 'usi_fk.parquet')
print('US/India: kind x grp')
print(u.group_by('fk', 'grp').len().pivot(on='grp', index='fk', values='len'))
f = u.filter(pl.col('fk') == 'fantasy')
print('US/India fantasy by country, numc: n, true rate, true rate among p>=.5, lowercase T / F')
print(f.group_by('country', 'numc').agg(n=pl.len(), tr=(pl.col('grp') == 'T').mean(), nacc=(pl.col('p') >= 0.5).sum(),
      tr_acc=(pl.col('grp') == 'T').filter(pl.col('p') >= 0.5).mean(),
      lowT=pl.col('low').filter(pl.col('grp') == 'T').mean(), lowF=pl.col('low').filter(pl.col('grp') != 'T').mean(),
      D=(pl.col('grp') == 'D').sum(), W=(pl.col('grp') == 'W').sum()).sort('country', 'n', descending=[False, True]))
print(f.sample(12, seed=5).select('sn', 'qn', 'grp', 'numc', 'p'))

fr = pl.read_parquet(TMP + 'france/fr_top.parquet', columns=['q', 's', 'qn', 'sn'])
fr = tag(fr)
fs = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc9', 'veto', 'numc', 'low', 'p2', 'bsig'])
fr = fr.join(fs, on=['q', 's'])
fr.select('q', 's', 'fk').write_parquet(F + 'fr_fk.parquet')
print('France: kind x accepted')
print(fr.group_by('fk').agg(n=pl.len(), acc=pl.col('acc9').sum(), veto=pl.col('veto').sum(),
                            low_acc=pl.col('low').filter(pl.col('acc9')).mean()).sort('n', descending=True))
ff = fr.filter(pl.col('fk') == 'fantasy')
print('France fantasy by numc')
print(ff.group_by('numc').agg(n=pl.len(), acc=pl.col('acc9').sum(), low_acc=pl.col('low').filter(pl.col('acc9')).mean(),
                              low_rej=pl.col('low').filter(~pl.col('acc9')).mean(), p2acc=pl.col('p2').filter(pl.col('acc9')).mean()).sort('n', descending=True))
print(ff.filter(pl.col('acc9')).sample(15, seed=5).select('sn', 'qn', 'numc', 'p2'))
