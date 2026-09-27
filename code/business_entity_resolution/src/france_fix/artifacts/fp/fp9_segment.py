"""Domain / handle names: can the inner string be written as a concatenation of S1 words (any order; each word full,
as its initial, accents transliterated or dropped; legal/noise words optional)? If not, the record's name holds a word
the S1 lacks = a second change inside the one-change form. US/India labels validate; France counts accepted."""
import sys, re, unicodedata
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from functools import lru_cache
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
F = f'{SCRATCH}/frfix2/fp/'
T = f'{SCRATCH}/'
DOM = re.compile(r'(?i)\.(com|net|org|fr|in|co|io|biz|info)(\.[a-z]{2})?\s*$')


def translit(x):
    return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower()


def dropacc(x):
    return ''.join(c for c in x.lower() if c.isascii())


def swords(sn):
    raw = [w for w in re.split(r"[^\w&+]+", sn or '') if w]
    out = set()
    for w in raw:
        for f in (translit(w), dropacc(w)):
            f = re.sub(r'[^a-z0-9]', '', f)
            if f:
                out.add(f)
    return out


def form(qn):
    if qn is None:
        return None, None
    s = qn.strip()
    if s[:1] in '#@' and ' ' not in s:
        return 'handle', re.sub(r'[^a-z0-9]', '', dropacc(s[1:]))
    if DOM.search(s) and ' ' not in s:
        return 'domain', re.sub(r'[^a-z0-9]', '', dropacc(DOM.sub('', s)))
    return None, None


def explained(inner, ws):
    """longest-prefix DP: inner = concat of words or single initials; returns number of unexplained chars (min)."""
    ws = [w for w in ws if w]
    inits = {w[0] for w in ws}
    n = len(inner)
    INF = 10 ** 9
    best = [INF] * (n + 1)   # min unexplained chars to cover inner[:i]
    best[0] = 0
    for i in range(n):
        if best[i] == INF:
            continue
        for w in ws:
            if inner.startswith(w, i):
                best[i + len(w)] = min(best[i + len(w)], best[i])
        if inner[i] in inits:
            best[i + 1] = min(best[i + 1], best[i])
        best[i + 1] = min(best[i + 1], best[i] + 1)      # unexplained char
    return best[n]


def check(qn, sn):
    k, inner = form(qn)
    if k is None:
        return None, None
    if not inner:
        return k, -1
    return k, explained(inner, swords(sn))


def tag(d):
    r = [check(a, b) for a, b in zip(d['qn'].to_list(), d['sn'].to_list())]
    return d.with_columns(form=pl.Series([x[0] for x in r], dtype=pl.Utf8), unexp=pl.Series([x[1] for x in r], dtype=pl.Int32))


FILT = r'^[#@]|\.[A-Za-z]{2,4}\s*$'
u = pl.read_parquet(T + 'france/usi_top.parquet', columns=['q', 's', 'label', 'p', 'qn', 'sn', 'country'])
u = tag(u.filter(pl.col('qn').str.contains(FILT))).filter(pl.col('form').is_not_null())
u = u.join(pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'numc']), on='q', how='left')
u = u.with_columns(ub=pl.col('unexp').cut([0, 2, 3], labels=['0', '1-2', '3', '4+']))
print('US/India: unexplained chars band x grp; true rate among p>=.5')
print(u.group_by('form', 'ub').agg(n=pl.len(), T=(pl.col('grp') == 'T').sum(), D=(pl.col('grp') == 'D').sum(),
      W=(pl.col('grp') == 'W').sum(), nacc=(pl.col('p') >= 0.5).sum(), Facc=((pl.col('grp') != 'T') & (pl.col('p') >= 0.5)).sum(),
      same=(pl.col('numc') == 'NSAME').mean(), up=pl.col('numc').is_in(['UP', 'UPbig']).mean()).sort('form', 'ub'))
print(u.filter(pl.col('unexp') >= 4).sample(12, seed=2).select('sn', 'qn', 'grp', 'numc', 'p', 'unexp'))

fr = pl.read_parquet(T + 'france/fr_top.parquet', columns=['q', 's', 'qn', 'sn'])
fr = tag(fr.filter(pl.col('qn').str.contains(FILT))).filter(pl.col('form').is_not_null())
fr = fr.join(pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc9', 'veto', 'numc', 'p2']), on=['q', 's'])
fr = fr.with_columns(ub=pl.col('unexp').cut([0, 2, 3], labels=['0', '1-2', '3', '4+']))
fr.write_parquet(F + 'fr_segment.parquet')
print('France: unexplained band')
print(fr.group_by('form', 'ub').agg(n=pl.len(), acc=pl.col('acc9').sum(), same=(pl.col('numc') == 'NSAME').mean(),
      up=pl.col('numc').is_in(['UP', 'UPbig']).mean(), acc_same=(pl.col('acc9') & (pl.col('numc') == 'NSAME')).sum(),
      acc_up=(pl.col('acc9') & pl.col('numc').is_in(['UP', 'UPbig'])).sum()).sort('form', 'ub'))
print(fr.filter((pl.col('unexp') >= 4) & pl.col('acc9')).sample(30, seed=1).select('sn', 'qn', 'numc', 'p2', 'unexp'))
