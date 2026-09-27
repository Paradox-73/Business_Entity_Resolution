# expected France macro F0.5 change vs v9b per variant: exact enumeration over each touched S1's changed pairs
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import sys
import itertools
import polars as pl
from common import WORK, id_to_int
R = f'{ROOT}/'; W = R + 'work/frfix2/'; B = f'{SCRATCH}/frfix2/build/'
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).filter(pl.col('country') == 'France')
NS1 = s1.height
def pairs(p):
    d = pl.read_csv(p, separator='\t', schema_overrides={'source1_entity_id': pl.Utf8, 'matched_entity_ids': pl.Utf8}, quote_char=None)
    d = d.filter(pl.col('source1_entity_id').is_in(s1['entity_id'].implode())).with_columns(q=pl.col('matched_entity_ids').fill_null('').str.split(','))
    return d.explode('q').filter(pl.col('q').fill_null('') != '').select(s=id_to_int('source1_entity_id'), q=id_to_int('q'))
base = pairs(R + 'work/frfix/out_descveto_gnadd_all/matching_results.tsv')
k0 = base.group_by('s').len('k0')
sc = pl.read_parquet(W + '../frfix/test_scores_frmin_descveto_gnadd.parquet', columns=['q', 's', 'p2'])
veto = pl.read_parquet(W + 'fp_veto_set.parquet', columns=['q', 's', 'tier']).with_columns(g=pl.lit('fp') + pl.col('tier'))
fl = pl.read_parquet(B + 'fn_clean_flags.parquet', columns=['q', 's', 'tier', 'lower', 'namb'])
fl = fl.with_columns(g=pl.when(pl.col('lower')).then(pl.lit('fnLOW')).otherwise(pl.lit('fn') + pl.col('tier')))
def f(tp, fn, fp):
    if tp + fn == 0: return float(fp == 0)
    return 0.0 if tp == 0 else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
def delta(rem, add, T, pe=0.7):
    """rem/add: (s, q, g, amb) rows; unchanged accepted pairs of the S1 are assumed true; an S1 left with no unchanged
    accepted pair has one hidden missed true record with prob 1-pe."""
    cast = lambda d: d.select('s', 'q', 'g', pl.col('amb').cast(pl.Int64), pl.col('p2').cast(pl.Float64))
    ch = pl.concat([cast(rem).with_columns(kind=pl.lit('r')), cast(add).with_columns(kind=pl.lit('a'))]).join(k0, on='s', how='left').with_columns(pl.col('k0').fill_null(0))
    tot = 0.0
    for (s,), grp in ch.group_by(['s']):
        rows = grp.to_dicts(); ku = rows[0]['k0'] - sum(r['kind'] == 'r' for r in rows)
        ts = [(T[r['g']] if r['g'] in T else r['p2']) / (1 + r['amb']) for r in rows]
        for hid, ph in ([(0, 1.0)] if ku > 0 else [(0, pe), (1, 1 - pe)]):
            for oc in itertools.product([0, 1], repeat=len(rows)):
                pr = ph
                for o, t in zip(oc, ts): pr *= t if o else 1 - t
                rt = sum(o for o, r in zip(oc, rows) if r['kind'] == 'r'); rf = sum(1 - o for o, r in zip(oc, rows) if r['kind'] == 'r')
                at = sum(o for o, r in zip(oc, rows) if r['kind'] == 'a'); af = sum(1 - o for o, r in zip(oc, rows) if r['kind'] == 'a')
                tot += pr * (f(ku + at, rt + hid, af) - f(ku + rt, at + hid, rf))
    return tot / NS1
def changes(out):
    P = pairs(out)
    rem = base.join(P, on=['s', 'q'], how='anti'); add = P.join(base, on=['s', 'q'], how='anti')
    lab = lambda d, L: (d.join(L.select('s', 'q', 'g', amb=(pl.col('namb') if 'namb' in L.columns else pl.lit(0))), on=['s', 'q'], how='left')
                         .join(sc, on=['s', 'q'], how='left').with_columns(pl.col('g').fill_null('knock'), pl.col('amb').fill_null(0)))
    return lab(rem, veto), lab(add, fl)
SC = {'central':     {'fpA': .14, 'fpB': .63, 'fpC': .50, 'fnA1': .92, 'fnA2': .90, 'fnA3': .90, 'fnA4': .90, 'fnLOW': .2},
      'pessimistic': {'fpA': .25, 'fpB': .82, 'fpC': .72, 'fnA1': .75, 'fnA2': .74, 'fnA3': .74, 'fnA4': .74, 'fnLOW': .2},
      'optimistic':  {'fpA': .00, 'fpB': .45, 'fpC': .45, 'fnA1': .95, 'fnA2': .95, 'fnA3': .95, 'fnA4': .92, 'fnLOW': .2}}
for v in ['fp', 'fpfn']:
    rem, add = changes(W + f'out_{v}_all/matching_results.tsv')
    print(v, 'removed', rem['g'].value_counts().sort('g').rows(), 'added', add['g'].value_counts().sort('g').rows())
    print('  knock-on p2: removed', rem.filter(pl.col('g') == 'knock')['p2'].describe().rows()[4:], 'added', add.filter(pl.col('g') == 'knock')['p2'].to_list())
    for nm, T in SC.items():
        d = delta(rem, add, T); dr = delta(rem, add.head(0), T); da = delta(rem.head(0), add, T)
        print(f'  {nm:12s} France {d:+.5f} (removals {dr:+.5f}, additions {da:+.5f}) -> LB {0.15 * d:+.6f}')
# what the extra fn cleanup (lowercase + sibling-ambiguous) is worth: those 208 pairs added on top of the final set
extra = fl.filter(pl.col('lower') | (pl.col('namb') > 0)).select('s', 'q', 'g', amb='namb').with_columns(p2=pl.lit(None, pl.Float32))
for nm, T in SC.items():
    print(f'  adding back the 208 dropped fn pairs ({nm}): France {delta(extra.head(0), extra, T):+.5f}')
