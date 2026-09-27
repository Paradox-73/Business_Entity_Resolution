# expected France macro-F0.5 change of adding the pairs, per touched S1 (k = current v9b count; each added pair true with prob t)
import polars as pl, numpy as np
from math import comb
NS1 = 259452
g = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/sim_added_c.parquet')
print(g.height, g.columns, g['c'].describe())
per = g.group_by('s').agg(k=pl.col('c').first(), m=pl.len(), tiers=pl.col('tier'))
print('S1 touched', per.height, 'm dist', per['m'].value_counts().sort('m').head(5).to_dicts())
def F(tp, fn, fp):
    if tp == 0 and fn == 0 and fp == 0: return 1.0
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
def edelta(k, m, t, prec=1.0, extra_fn=0.0):
    # current k predicted: kt = k*prec true, k*(1-prec) false (expected values used), extra_fn other misses
    kt, kf = k * prec, k * (1 - prec)
    e = 0.0
    for x in range(m + 1):
        pr = comb(m, x) * t ** x * (1 - t) ** (m - x)
        e += pr * (F(kt + x, extra_fn, kf + m - x) - F(kt, x + extra_fn, kf))
    return e
ks, ms = per['k'].to_list(), per['m'].to_list()
for prec, xfn in [(1.0, 0.0), (0.95, 0.15)]:
    row = []
    for t in [0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95]:
        row.append((t, round(sum(edelta(k, m, t, prec, xfn) for k, m in zip(ks, ms)) / NS1, 5)))
    print('prec', prec, 'extra_fn', xfn, row)
# tier split at given t per tier
tiers = {'A1': 0.85, 'A2': 0.93, 'A3': 0.85, 'A4': 0.88}
