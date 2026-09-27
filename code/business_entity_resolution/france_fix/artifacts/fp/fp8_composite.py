"""Composite name forms (web domain 'x.com', social handle '#x'/'@x', squashed): the form is itself the record's one change,
so the inner text should equal the S1 name squashed. Inner mismatch = a second change (decoy word) hidden in the form.
Validate on US/India labels, then count France accepted pairs with inner mismatch."""
import sys, re, unicodedata
import polars as pl
sys.path.insert(0, 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census')
from ops import LEGAL, NOISE, TITLE, STOPW
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
T = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/'
DOM = re.compile(r'(?i)\.(com|net|org|fr|in|co|io|biz|info)(\.[a-z]{2})?\s*$')


def sa(x):
    return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower()


def wl(x):
    x = sa(x)
    x = re.sub(r'\b((?:[a-z]\.){2,})', lambda m: m.group(1).replace('.', ''), x)
    return [w for w in re.split(r'[^a-z0-9&+]+', x) if w]


def form(qn):
    if qn is None:
        return None, None
    s = qn.strip()
    if s[:1] in '#@' and ' ' not in s:
        return 'handle', re.sub(r'[^a-z0-9]', '', sa(s[1:]))
    if DOM.search(s) and ' ' not in s.strip():
        return 'domain', re.sub(r'[^a-z0-9]', '', sa(DOM.sub('', s)))
    return None, None


def variants(sn):
    w = wl(sn or '')
    core = [x for x in w if x not in LEGAL]
    v = set()
    for ws in (w, core, [x for x in core if x not in NOISE and x not in ('&', '+')],
               [x for x in core if x not in NOISE and x not in ('&', '+', 'freres', 'soeurs')],
               [x for x in core if x not in STOPW], [x for x in core if x not in TITLE]):
        j = ''.join(ws).replace('&', '').replace('+', '')
        if j:
            v.add(j)
    # also first word / first two words (truncation)
    if core:
        v.add(core[0])
    return v


def check(qn, sn):
    k, inner = form(qn)
    if k is None:
        return None, None
    if not inner:
        return k, 'empty'
    v = variants(sn)
    if inner in v:
        return k, 'exact'
    if any(inner.startswith(x) or x.startswith(inner) for x in v if len(x) >= 3):
        return k, 'prefix'
    return k, 'mismatch'


def tag(d):
    r = [check(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())]
    return d.with_columns(form=pl.Series([x[0] for x in r], dtype=pl.Utf8), inner=pl.Series([x[1] for x in r], dtype=pl.Utf8))


u = pl.read_parquet(T + 'france/usi_top.parquet', columns=['q', 's', 'label', 'p', 'qn', 'sn', 'country'])
u = tag(u.filter(pl.col('qn').str.contains(r'^[#@]|\.[A-Za-z]{2,4}\s*$'))).filter(pl.col('form').is_not_null())
g = pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'numc'])
u = u.join(g, on='q', how='left')
print('US/India composite forms: counts by form, inner, grp; true rate; true rate among p>=.5')
print(u.group_by('country', 'form', 'inner').agg(n=pl.len(), T=(pl.col('grp') == 'T').sum(), D=(pl.col('grp') == 'D').sum(),
      W=(pl.col('grp') == 'W').sum(), nacc=(pl.col('p') >= 0.5).sum(), tr_acc=(pl.col('grp') == 'T').filter(pl.col('p') >= 0.5).mean(),
      same=(pl.col('numc') == 'NSAME').mean()).sort('country', 'form', 'inner'))
print(u.filter(pl.col('inner') == 'mismatch').sample(12, seed=1).select('sn', 'qn', 'grp', 'numc', 'p'))

fr = pl.read_parquet(T + 'france/fr_top.parquet', columns=['q', 's', 'qn', 'sn'])
fr = tag(fr.filter(pl.col('qn').str.contains(r'^[#@]|\.[A-Za-z]{2,4}\s*$'))).filter(pl.col('form').is_not_null())
fs = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc9', 'veto', 'numc', 'p2'])
fr = fr.join(fs, on=['q', 's'])
fr.write_parquet(F + 'fr_composite.parquet')
print('France composite forms')
print(fr.group_by('form', 'inner').agg(n=pl.len(), acc=pl.col('acc9').sum(), veto=pl.col('veto').sum(), same=(pl.col('numc') == 'NSAME').mean(),
      up=pl.col('numc').is_in(['UP', 'UPbig']).mean(), same_acc=((pl.col('numc') == 'NSAME') & pl.col('acc9')).sum()).sort('form', 'inner'))
print(fr.filter((pl.col('inner') == 'mismatch') & pl.col('acc9')).sample(25, seed=1).select('sn', 'qn', 'numc', 'p2'))
