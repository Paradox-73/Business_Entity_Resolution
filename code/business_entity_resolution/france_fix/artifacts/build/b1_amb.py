# strict sibling ambiguity + lowercase check for the cleaned fn add set
import sys
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
import polars as pl
from ops import words, undot_legal, LEGAL, NOISE, STOPW, house, LEET
from rapidfuzz import fuzz
def _match(a, b):   # stricter than census _match (75): 'nantes' ~ 'enfants' is 77
    return a == b or (len(a) > 3 and len(b) > 3 and fuzz.ratio(a, b) >= 85) or a.translate(LEET) == b.translate(LEET)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
FN = 'C:/ber_scratch/frfix2/fn/'
OUT = 'C:/ber_scratch/frfix2/build/'
A = pl.read_parquet(FN + 'fn_add_all.parquet', columns=['q', 's', 'tier', 'lower', 'qn', 'sn', 'qa', 'sa'])
cl = pl.read_parquet('E:/Projects/Amazon ML Challenge/work/frfix2/fn_add_set_clean.parquet').with_columns(clean=pl.lit(True))
A = A.join(cl, on=['q', 's'], how='inner')
tp = pl.read_parquet(FN + 'refute/fn_twin_pairs.parquet', columns=['q', 's', 's2', 'n2', 'a2'])
x = tp.join(A.select('q', 's', 'tier', 'qn', 'sn', 'qa', 'sa'), on=['q', 's'], how='inner')
def content(n): return [w for w in words(undot_legal(n or '')) if w not in LEGAL and w not in NOISE and w not in STOPW and w not in ('&', '+')]
def legal(n): return sorted(set(w for w in words(undot_legal(n or '')) if w in LEGAL))
def um(a, b):
    b = list(b); u = 0
    for w in a:
        j = next((j for j, v in enumerate(b) if _match(w, v)), None)
        if j is None: u += 1
        else: b.pop(j)
    return u
def hn(a):
    h = house(a or '', 'France'); return h[0] if h else None
def allw(n): return [w for w in words(undot_legal(n or '')) if w not in LEGAL and w not in STOPW and w not in ('&', '+')]
def score(qn, n):
    miss = um(content(qn), content(n))                  # record content words (not noise/legal) the name cannot explain
    a, b = allw(qn), allw(n)
    edits = max(um(a, b), um(b, a))                     # word edits, a swap counts once
    lpen = int(legal(qn) != legal(n))                   # legal form added / dropped / changed
    return miss, edits, lpen
rows = []
for r in x.iter_rows(named=True):
    m1, e1, l1 = score(r['qn'], r['sn']); m2, e2, l2 = score(r['qn'], r['n2'])
    hq, h1, h2 = hn(r['qa']), hn(r['sa']), hn(r['a2'])
    num_ok = (hq is None) or (h2 == hq)          # record with a number: the sibling must carry the same number
    amb = num_ok and m2 <= m1 and (m2 + e2 + l2) <= (m1 + e1 + l1)
    rows.append((r['q'], r['s'], r['s2'], amb, m1, e1, l1, m2, e2, l2, hq, h1, h2))
S = pl.DataFrame(rows, schema=['q', 's', 's2', 'amb', 'm1', 'e1', 'l1', 'm2', 'e2', 'l2', 'hq', 'h1', 'h2'], orient='row')
per = S.group_by('q', 's').agg(namb=pl.col('amb').sum())
A = A.join(per, on=['q', 's'], how='left').with_columns(pl.col('namb').fill_null(0))
print(A.group_by('tier').agg(n=pl.len(), low=pl.col('lower').sum(), amb=(pl.col('namb') > 0).sum(),
      exp_wrong=(pl.col('namb') / (pl.col('namb') + 1)).sum()).sort('tier'))
ex = S.filter('amb').join(x.select('q', 's', 's2', 'tier', 'qn', 'sn', 'n2', 'qa', 'a2'), on=['q', 's', 's2'])
print(ex.sample(min(25, ex.height), seed=5).select('tier', 'qn', 'sn', 'n2', 'hq', 'h1', 'h2'))
nonamb = S.filter(~pl.col('amb')).join(x.select('q', 's', 's2', 'tier', 'qn', 'sn', 'n2'), on=['q', 's', 's2'])
print('rejected siblings sample'); print(nonamb.sample(min(12, nonamb.height), seed=5).select('tier', 'qn', 'sn', 'n2', 'm1', 'e1', 'm2', 'e2', 'hq', 'h2'))
A.select('q', 's', 'tier', 'lower', 'namb', 'qn', 'sn', 'qa', 'sa').write_parquet(OUT + 'fn_clean_flags.parquet')
