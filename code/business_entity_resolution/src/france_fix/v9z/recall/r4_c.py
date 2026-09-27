# (c) Rejected single-change copies at the same address with only-true changes (acronym, web domain, dotted legal form):
# name fit test + sibling ambiguity (another S1 at the same address that fits equally).
import sys, re, unicodedata
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from rapidfuzz import fuzz
from ops import words, undot_legal, LEGAL
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(330); pl.Config.set_fmt_str_lengths(45)
R = f'{SCRATCH}/frfix3/recall/'
TW = f'{SCRATCH}/frfix/twins/'
d = pl.read_parquet(R + 'rej_sameaddr.parquet')
keys = ['DOMAIN||NSAME', 'DOMAIN||NMISS', 'ACRONYM||NSAME', 'ACRONYM+LEGAL_DROP||NSAME', 'ACRONYM+LEGAL_DROP||NMISS', 'ACRONYM||NMISS',
        'LEGAL_DOT||NMISS', 'LEGAL_DOT||NSAME']
d = d.filter(pl.col('key').is_in(keys))
rk = pl.read_parquet(TW + 'fr_recsk.parquet', columns=['q', 'qnum', 'qst', 'qcity'])
s1 = pl.read_parquet(TW + 'fr_s1k.parquet')
d = d.join(rk, on='q', how='left')
STOP = set('de des du d l la le les et en a au aux'.split())
def drop_acc(x):   # accented letters removed entirely (generator's domain builder drops them)
    return ''.join(ch for ch in (x or '') if ord(ch) < 128)
def strip(x):
    return unicodedata.normalize('NFKD', x or '').encode('ascii', 'ignore').decode()
def initials(n):
    w = [t for t in words(undot_legal(n or '')) if t not in STOP]
    return ''.join(t[0] for t in w), ''.join(t[0] for t in w if t not in LEGAL)
def acr_fit(qn, sn):
    a = re.sub(r'[^a-z]', '', strip(qn).lower())
    i1, i2 = initials(sn)
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
    b = re.sub(r'\.(com|fr|net|org|eu|info|biz)$', '', (qn or '').lower().strip())
    b = re.sub(r'[^a-z0-9]', '', b)
    return b != '' and max((fuzz.ratio(b, f) for f in dom_forms(sn)), default=0) >= 92
def fit(key, qn, sn):
    if key.startswith('ACRONYM'): return acr_fit(qn, sn)
    if key.startswith('DOMAIN'): return dom_fit(qn, sn)
    return True
d = d.with_columns(fit=pl.Series([fit(k, a, b) for k, a, b in zip(d['key'], d['qn'], d['sn'])]))
# siblings: other S1 at the same address (number records) or same street (number missing) that also fit
sib = d.select('q', 's', 'key', 'qn', 'qnum', 'qst', 'qcity').join(s1.select(s2='s', n2='sn', snum='snum', qst='sst', qcity='scity'), on=['qcity', 'qst'])
sib = sib.filter((pl.col('s2') != pl.col('s')) & ((pl.col('qnum') == pl.col('snum')) | pl.col('qnum').is_null() | (pl.col('qnum') == '')))
sib = sib.with_columns(f2=pl.Series([fit(k, a, b) if k[:3] in ('ACR', 'DOM') else (fuzz.token_sort_ratio(strip(undot_legal(a)).lower(), strip(undot_legal(b)).lower()) >= 90)
                                     for k, a, b in zip(sib['key'], sib['qn'], sib['n2'])]))
amb = sib.group_by('q', 's').agg(nsib=pl.len(), namb=pl.col('f2').sum())
d = d.join(amb, on=['q', 's'], how='left').with_columns(pl.col('nsib', 'namb').fill_null(0))
print(d.group_by('key', 'fit', pl.col('namb').clip(0, 1).alias('amb')).agg(n=pl.len(), g=pl.col('p2g').median(), t=pl.col('p3').median(), p22=(pl.col('p2_2') > 0.3).sum()).sort('key', 'fit', 'amb'))
d.write_parquet(R + 'c_cands.parquet')
ok = d.filter(pl.col('fit') & (pl.col('namb') == 0))
for k in ok['key'].unique().sort().to_list():
    x = ok.filter(pl.col('key') == k)
    print(k, x.height); print(x.sample(min(8, x.height), seed=4).select('qn', 'sn', 'qa', 'sa', 'p2g', 'p3', 'nsib'))
bad = d.filter(~pl.col('fit') & pl.col('key').str.starts_with('DOMAIN'))
print('domain not fitting sample'); print(bad.sample(8, seed=4).select('qn', 'sn', 'p2g', 'p3'))
