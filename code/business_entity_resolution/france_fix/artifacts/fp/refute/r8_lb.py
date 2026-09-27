"""LB history overlap: veto pairs vs v7ens / v7g_num / v7j / v7m / v7i pair sets; lowercase test of v7g_num's removals."""
import polars as pl
from scipy.stats import chi2
T = 'C:/ber_scratch/'
F = T + 'frfix2/fp/'
v = pl.read_parquet(F + 'fp_veto_set.parquet', columns=['q', 's', 'tier'])
E = pl.read_parquet(T + 'france/pairs_v7ens.parquet')
print('pairs file cols', E.columns, E.height)
print('veto pairs in v7ens accepted:', v.join(E, on=['s', 'q']).height)
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'acc9', 'numc', 'low', 'base', 'nchg'])
nm = pl.col('base').str.split('+')
fr = fr.with_columns(A=(pl.col('numc') == 'UP') & nm.list.contains('LEGAL_ADD'))
sets = {'v7g_num': T + 'france/pairs_v7g_num.parquet', 'v7j': T + 'france/pairs_v7j.parquet', 'v7i': T + 'france/pairs_v7i.parquet', 'v7m': T + 'frfix/namechg/pairs_v7m.parquet'}
for k, p in sets.items():
    V = pl.read_parquet(p)
    add = V.join(E, on=['s', 'q'], how='anti'); rem = E.join(V, on=['s', 'q'], how='anti')
    vr = v.join(rem, on=['s', 'q']); va = v.join(add, on=['s', 'q'])
    ra = rem.join(fr, on=['q', 's'])
    aa = add.join(fr, on=['q', 's'])
    print(f'{k}: added {add.height}, removed {rem.height}; veto pairs removed {vr.height} {vr.group_by("tier").len().sort("tier").to_dicts()}, veto pairs added {va.height}')
    print(f'   removed with tier-A signature: {ra.filter(pl.col("A")).height} (lowercase {ra.filter(pl.col("A"))["low"].sum()}); added tier-A signature {aa.filter(pl.col("A")).height} (lowercase {aa.filter(pl.col("A"))["low"].mean() if aa.filter(pl.col("A")).height else None})')
