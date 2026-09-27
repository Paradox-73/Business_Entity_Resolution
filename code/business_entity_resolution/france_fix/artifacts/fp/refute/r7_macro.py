"""Macro F0.5 effect of the veto: k = accepted records of the S1 in v9b; US/India share of decoy-targeted S1 rows with no
true record; France expected delta for tier true-rate scenarios; micro check."""
import os, sys
sys.path.insert(0, "E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int, read_truth
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
R = F + 'refute/'
tr = read_truth().select(s=id_to_int('s1_id'), q=id_to_int('q_id'))
nt = tr.group_by('s').len('nt')
u = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france/usi_top.parquet', columns=['q', 's', 'label', 'p', 'country'])
sig = pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'base', 'numc'])
u = u.join(sig, on='q').join(nt, on='s', how='left').with_columns(pl.col('nt').fill_null(0))
# accepted-set k per s in US/India (p >= .5 best candidates) approximates the decision
k = u.filter(pl.col('p') >= 0.5).group_by('s').len('k')
u = u.join(k, on='s', how='left').with_columns(pl.col('k').fill_null(0))
x = u.filter((pl.col('grp') != 'T') & (pl.col('numc') == 'UP') & (pl.col('base').str.contains('LEGAL_ADD')))
print('US/India decoy LEGAL_ADD*|UP: share of their best S1 with no true record, all / p>=.5 & k==1:',
      round((x['nt'] == 0).mean(), 3), round((x.filter((pl.col('p') >= 0.5) & (pl.col('k') == 1))['nt'] == 0).mean(), 3),
      x.filter((pl.col('p') >= 0.5) & (pl.col('k') == 1)).height)
x = u.filter((pl.col('grp') != 'T') & (pl.col('p') >= 0.5))
print('US/India all accepted-like false best candidates: share S1 empty when k==1:', round((x.filter(pl.col('k') == 1)['nt'] == 0).mean(), 3), x.filter(pl.col('k') == 1).height)
s1 = pl.read_parquet(os.path.join(WORK, 'test_s1.parquet'), columns=['entity_id', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'))
n_s1 = s1.height
acc = pl.read_parquet(F + 'acc_v9b.parquet', columns=['s', 'q'])
ka = acc.group_by('s').len('k')
v = pl.read_parquet(F + 'fp_veto_set.parquet', columns=['q', 's', 'tier']).join(ka, on='s', how='left')
print('France S1 rows', n_s1, '; S1 with >= 1 accepted', ka.height, '; veto k distribution by tier:')
print(v.group_by('tier', 'k').len().sort('tier', 'k').pivot(on='k', index='tier', values='len'))
print('S1 rows hit by >1 vetoed pair:', v.group_by('s').len().filter(pl.col('len') > 1).height)
def f05(p, r):
    return 0.0 if p == 0 or r == 0 else 1.25 * p * r / (0.25 * p + r)
def pair(kk, t, pe):
    if kk == 1:
        g, l = pe, 1.0
    else:
        g, l = 1 - f05((kk - 1) / kk, 1.0), 1 - f05(1.0, (kk - 1) / kk)
    return (1 - t) * g - t * l
for scen, ts in {'low (A .0, B .45, C .45)': {'A': 0.0, 'B': 0.45, 'C': 0.45},
                 'mid (A .14, B .63, C .5)': {'A': 0.14, 'B': 0.63, 'C': 0.5},
                 'high (A .25, B .82, C .72)': {'A': 0.25, 'B': 0.82, 'C': 0.72}}.items():
    for pe in (0.5, 0.9):
        m = {t: sum(pair(a, ts[t], pe) for a in v.filter(pl.col('tier') == t)['k'].to_list()) / n_s1 for t in 'ABC'}
        print(f'{scen} p_empty {pe}: macro France delta A {m["A"]:+.5f} B {m["B"]:+.5f} C {m["C"]:+.5f} total {sum(m.values()):+.5f} -> LB {0.15 * sum(m.values()):+.6f}')
# knock-on: 1 removal p2 .757, 7 additions p2 .53-.72 (assume true prob = p2)
rem = pl.read_parquet(R + 'removed.parquet'); add = pl.read_parquet(R + 'added.parquet')
sc = pl.read_parquet(F + 'fr_scores_v9b_lv.parquet', columns=['q', 's', 'p2'])
add = add.join(sc, on=['q', 's'], how='left').join(ka, on='s', how='left').with_columns(pl.col('k').fill_null(0))
print('knock-on additions: k of their S1 in v9b', add.select('p2', 'k').to_dicts())
