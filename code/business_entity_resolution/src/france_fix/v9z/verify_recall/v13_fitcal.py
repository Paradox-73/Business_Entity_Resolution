# Calibrate the (c) name-fit filters on US/India labels: truth rate of DOMAIN / ACRONYM / LEGAL_DOT pairs by model band and fit
import sys, re, unicodedata
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from rapidfuzz import fuzz
from ops import words, undot_legal, LEGAL
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45); pl.Config.set_float_precision(3)
C = f'{SCRATCH}/frfix2/census/'
STOP = set('de des du d l la le les et en a au aux'.split())
def drop_acc(x): return ''.join(ch for ch in (x or '') if ord(ch) < 128)
def strip(x): return unicodedata.normalize('NFKD', x or '').encode('ascii', 'ignore').decode()
def initials(n):
    w = [t for t in words(undot_legal(n or '')) if t not in STOP]
    return ''.join(t[0] for t in w), ''.join(t[0] for t in w if t not in LEGAL)
def acr_fit(qn, sn):
    a = re.sub(r'[^a-z]', '', strip(qn).lower()); i1, i2 = initials(sn)
    return a != '' and (a == i1 or a == i2)
def dom_forms(sn):
    out = set()
    for f in (strip, drop_acc):
        w = [t for t in re.split(r'[^a-z0-9]+', f(sn).lower()) if t]
        wl = [t for t in w if t not in LEGAL]
        for ws in (w, wl, [t for t in w if t not in STOP], [t for t in wl if t not in STOP]):
            if ws:
                out.add(''.join(ws)); out.add(ws[0][0] + ''.join(ws[1:]))
                if len(ws) > 1:
                    out.add(''.join(ws[1:])); out.add(''.join(t[0] for t in ws[:-1]) + ws[-1])
    return out
def dom_fit(qn, sn):
    b = re.sub(r'\.(com|fr|net|org|eu|info|biz)$', '', (qn or '').lower().strip()); b = re.sub(r'[^a-z0-9]', '', b)
    return b != '' and max((fuzz.ratio(b, f) for f in dom_forms(sn)), default=0) >= 92
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'), bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                   wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = u.with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'))
keys = ['DOMAIN||NSAME', 'DOMAIN||NMISS', 'ACRONYM||NSAME', 'ACRONYM+LEGAL_DROP||NSAME', 'ACRONYM+LEGAL_DROP||NMISS', 'ACRONYM||NMISS', 'LEGAL_DOT||NMISS', 'LEGAL_DOT||NSAME']
x = u.filter(pl.col('key').is_in(keys)).join(pl.read_parquet(C + 'base.parquet', columns=['q', 'qn', 'sn']), on='q')
def fit(k, a, b):
    if k.startswith('ACRONYM'): return acr_fit(a, b)
    if k.startswith('DOMAIN'): return dom_fit(a, b)
    return True
x = x.with_columns(fit=pl.Series([fit(k, a, b) for k, a, b in zip(x['key'], x['qn'], x['sn'])]), T=pl.col('grp') == 'T',
                   band=pl.when(pl.col('p') < 0.05).then(pl.lit('a<.05')).when(pl.col('p') < 0.8).then(pl.lit('b.05-.8')).otherwise(pl.lit('c>=.8')))
print(x.group_by('key', 'band', 'fit').agg(n=pl.len(), tr=pl.col('T').mean(), D=(pl.col('grp') == 'D').sum(), W=(pl.col('grp') == 'W').sum()).sort('key', 'band', 'fit'))
print(x.filter(pl.col('key').str.starts_with('DOMAIN') & pl.col('fit') & (pl.col('p') < 0.8)).select('qn', 'sn', 'grp', 'p').head(25))
