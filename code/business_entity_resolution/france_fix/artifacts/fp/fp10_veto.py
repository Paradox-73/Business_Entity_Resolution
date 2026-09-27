"""Build the France false-accept veto on top of v9b, estimate its true rate (lowercase test) and the France gain,
write the scores file (v9b scores, France p2 = 0 for vetoed pairs) and check the resulting France decision."""
import os, sys
import numpy as np
import polars as pl
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
from pipeline import decide_expf
from fp1_dec import france_scores, V9B
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(50)
F = 'C:/ber_scratch/frfix2/fp/'
T = 'C:/ber_scratch/'
OUTW = 'E:/Projects/Amazon ML Challenge/work/frfix2/'

fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc9', 'veto', 'num', 'numc', 'low', 'base', 'bsig', 'nchg', 'p2'])
ref = pl.read_parquet(F + 'ref_bsig.parquet').select('bsig', 'lowT', 'lowF', 'nT', 'nF', 'tr', 'tr_acc')
fr = fr.join(ref, on='bsig', how='left')
nm = pl.col('base').str.split('+')
hasLA = nm.list.contains('LEGAL_ADD')
hasD = nm.list.eval(pl.element().is_in(['ADD', 'SWAP', 'TYPO', 'LEGAL_CHANGE'])).list.any()
hasDrop = nm.list.eval(pl.element().is_in(['DROP', 'DUP'])).list.any()
UP = pl.col('numc') == 'UP'
fr = fr.with_columns(
    tA=UP & hasLA,
    tB=UP & ~hasLA & hasD & ~hasDrop,
    # C: record lowercase + a name change, in signatures where US/India true records are (almost) never lowercase
    tC=(pl.col('low') & (pl.col('nchg') >= 1) & (pl.col('lowT') <= 0.002) & (pl.col('lowF') >= 0.02)
        & (pl.col('nT') >= 300) & ~UP))

# lowercase-based true-rate estimate per tier: accepted share vs France rejected share of the same tier (decoy ref)
def est(mask, name):
    a = fr.filter(pl.col('acc9') & mask)
    r = fr.filter(~pl.col('acc9') & ~pl.col('veto') & mask)
    s, sr = a['low'].mean(), r['low'].mean()
    lt = a['lowT'].mean()
    se = np.sqrt(s * (1 - s) / a.height)
    t = 1 - (s - lt) / (sr - lt)
    print(f'{name}: accepted {a.height}, lowercase {s:.4f} (+-{se:.4f}); France rejected same tier {r.height} lowercase {sr:.4f}; '
          f'US/India true-lowercase {lt:.4f} -> true share {t:.2f} (1 SE range {1 - (s + se - lt) / (sr - lt):.2f}..{1 - (s - se - lt) / (sr - lt):.2f})')
    return t

tA = est(pl.col('tA'), 'A legal form added + number up 1-20')
tB = est(pl.col('tB'), 'B word added/swapped/typo + number up 1-20')
# C: per-pair P(true | lowercase) = (1-f) lowT / s, f from the group's lowercase share
g = (fr.filter(pl.col('acc9') & (pl.col('nchg') >= 1)).group_by('bsig')
       .agg(n=pl.len(), s=pl.col('low').mean(), lowT=pl.col('lowT').first(), lowF=pl.col('lowF').first()))
g = g.with_columns(f=((pl.col('s') - pl.col('lowT')) / (pl.col('lowF') - pl.col('lowT'))).clip(0, 1)).with_columns(
    pt_low=((1 - pl.col('f')) * pl.col('lowT') / pl.col('s')).clip(0, 1))
C = fr.filter(pl.col('acc9') & pl.col('tC')).join(g.select('bsig', 'pt_low', 'f'), on='bsig')
print('C: lowercase + name change, by signature (pt_low = P(true | lowercase))')
print(C.group_by('bsig').agg(n=pl.len(), pt_low=pl.col('pt_low').first(), f=pl.col('f').first()).sort('n', descending=True))
C = C.filter(pl.col('pt_low') < 0.6)
tC = C['pt_low'].mean() if C.height else 0.0

V = pl.concat([fr.filter(pl.col('acc9') & pl.col('tA')).select('q', 's', 'bsig', 'num', 'low', 'p2').with_columns(tier=pl.lit('A'), t=pl.lit(max(tA, 0.0))),
               fr.filter(pl.col('acc9') & pl.col('tB')).select('q', 's', 'bsig', 'num', 'low', 'p2').with_columns(tier=pl.lit('B'), t=pl.lit(max(tB, 0.0))),
               C.select('q', 's', 'bsig', 'num', 'low', 'p2').with_columns(tier=pl.lit('C'), t=C['pt_low'])]).unique(['q', 's'], keep='first')
print('VETO:', V.height, V.group_by('tier').agg(n=pl.len(), t=pl.col('t').mean()).sort('tier').to_dicts())
top = pl.read_parquet(T + 'france/fr_top.parquet', columns=['q', 's', 'qn', 'sn', 'qa', 'sa'])
V = V.join(top, on=['q', 's'], how='left')
os.makedirs(OUTW, exist_ok=True)
V.write_parquet(OUTW + 'fp_veto_set.parquet')
V.write_parquet(F + 'fp_veto_set.parquet')

# ---- expected France gain: micro approximation (team's) and macro with each S1's accepted count k
acc = pl.read_parquet(F + 'acc_v9b.parquet', columns=['s', 'q'])
k = acc.group_by('s').len('k')
V = V.join(k, on='s', how='left')
n_s1 = pl.read_parquet('E:/Projects/Amazon ML Challenge/work/test_s1.parquet', columns=['country']).filter(pl.col('country') == 'France').height
F0, D0 = 0.974, 1.07e6          # v9b France micro F0.5 and denominator (1.25 TP + 0.25 FN + FP), see notes
micro = float(((F0 - 1.25 * V['t']) / D0).sum())


def f05(p, r):
    return 0.0 if p == 0 or r == 0 else 1.25 * p * r / (0.25 * p + r)


def macro_pair(kk, t, p_empty):
    # fake: other kk-1 accepted records true, S1 has no other true records (full recall)
    if kk == 1:
        gain_fake = p_empty          # np 1, tp 0 -> np 0: F 0 -> 1 if the S1 has no true records
        loss_true = 1.0
    else:
        gain_fake = 1 - f05((kk - 1) / kk, 1.0)
        loss_true = 1 - f05(1.0, (kk - 1) / kk)
    return (1 - t) * gain_fake - t * loss_true


for pe in (0.5, 1.0):
    mac = sum(macro_pair(a, b, pe) for a, b in zip(V['k'].to_list(), V['t'].to_list())) / n_s1
    print(f'macro France gain (P(S1 empty | lone fake)={pe}): {mac:+.5f}')
print(f'micro France gain: {micro:+.5f};  France S1 rows {n_s1};  k distribution {V["k"].value_counts().sort("k").head(8).to_dicts()}')

# ---- scores file: v9b with France p2 = 0 for the veto
b = pl.read_parquet(V9B)
vv = V.select('q', 's').with_columns(_v=pl.lit(True))
b = b.join(vv, on=['q', 's'], how='left').with_columns(
    p2=pl.when(pl.col('_v').fill_null(False)).then(0.0).otherwise(pl.col('p2')).cast(pl.Float32)).drop('_v')
out = OUTW + 'test_scores_v9b_fpveto.parquet'
b.write_parquet(out)
print('wrote', out, b.height, 'rows; p2 set to 0 on', V.height)
del b
# France decision check
nb = france_scores(out)
m = decide_expf(nb, 'p2', 0.5, 1.0)
old = pl.read_parquet(F + 'acc_v9b.parquet', columns=['s', 'q'])
rem = old.join(m, on=['s', 'q'], how='anti')
new = m.join(old, on=['s', 'q'], how='anti')
print('new France decision:', m.height, 'pairs; removed vs v9b', rem.height, '(in veto', rem.join(V.select('s', 'q'), on=['s', 'q']).height,
      '); added vs v9b', new.height)
