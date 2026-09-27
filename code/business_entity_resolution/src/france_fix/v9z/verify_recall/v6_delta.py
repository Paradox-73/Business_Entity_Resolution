# Expected France macro-F0.5 change of a pair-set change vs a base, per S1 exact (common pairs true, no other missing records),
# added pairs true w.p. a, removed pairs true w.p. r (independent). Exact expectation via binomial sums per S1.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
from math import comb
V = f'{SCRATCH}/frfix3/verify_recall/'
NS = 259452
def F(tp, npred, ntrue):
    if npred == 0 and ntrue == 0: return 1.0
    if tp == 0: return 0.0
    P, R = tp / npred, tp / ntrue
    return 1.25 * P * R / (0.25 * P + R)
def per_s1(old, new):
    c = old.join(new, on=['s', 'q']).group_by('s').agg(c=pl.len())
    A = new.join(old, on=['s', 'q'], how='anti').group_by('s').agg(na=pl.len())
    Rm = old.join(new, on=['s', 'q'], how='anti').group_by('s').agg(nr=pl.len())
    s = pl.concat([c.select('s'), A.select('s'), Rm.select('s')]).unique()
    s = s.join(c, on='s', how='left').join(A, on='s', how='left').join(Rm, on='s', how='left').fill_null(0)
    s = s.filter((pl.col('na') > 0) | (pl.col('nr') > 0))
    return s.group_by('c', 'na', 'nr').agg(n=pl.len())
def dF(tab, a, r):
    tot = 0.0
    for c, na, nr, n in tab.iter_rows():
        e = 0.0
        for x in range(na + 1):
            px = comb(na, x) * a ** x * (1 - a) ** (na - x)
            for y in range(nr + 1):
                py = comb(nr, y) * r ** y * (1 - r) ** (nr - y)
                ntrue = c + x + y
                f_old = F(c + y, c + nr, ntrue)
                f_new = F(c + x, c + na, ntrue)
                e += px * py * (f_new - f_old)
        tot += n * e
    return tot / NS
if __name__ == '__main__':
    P = {v: pl.read_parquet(V + f'fr_{v}.parquet') for v in ['v7ens', 'v7m', 'v7i', 'v7j', 'v9e', 'v9f', 'v9y']}
    for v in ['v7i', 'v7m', 'v7j']:
        tab = per_s1(P['v7ens'], P[v])
        print(v, 'S1 touched', tab['n'].sum(), ' '.join(f'a={a:.2f}:{dF(tab, a, 0.9):+.5f}' for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
    tab = per_s1(P['v7ens'], P['v9y'])
    print('v9y vs v7ens: removed-true r / added-true a grid')
    for r in (0.0, 0.05, 0.1, 0.15, 0.2):
        print(f'  r={r:.2f}', ' '.join(f'a={a:.2f}:{dF(tab, a, r):+.5f}' for a in (0.5, 0.7, 0.8, 0.9, 0.96, 1.0)))
    add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
    new = pl.concat([P['v9y'], add.select('s', 'q')])
    tab = per_s1(P['v9y'], new)
    print('recall adds only (no drops) vs v9y: S1 touched', tab['n'].sum(), 'adds per S1 dist', tab.group_by('na').agg(pl.col('n').sum()).sort('na').rows())
    print('  c dist of touched S1', tab.group_by('c').agg(pl.col('n').sum()).sort('c').rows())
    print(' ', ' '.join(f'a={a:.2f}:{dF(tab, a, 1.0):+.5f}' for a in (0.6, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 1.0)))
