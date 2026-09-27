# direction of the swapped word pair (S1 word class -> record word class) for n_swap:noise signatures: US/India true/false vs France tiers
import sys
sys.path.insert(0, 'C:/ber_scratch/frfix2/census')
import polars as pl
from ops import words, undot_legal, LEGAL, wclass, _match
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_float_precision(4)
def swaps(qn, sn):
    qx = [w for w in words(undot_legal(qn or '')) if w not in LEGAL]; sx = [w for w in words(undot_legal(sn or '')) if w not in LEGAL]
    used = [False] * len(qx); us = []
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None: j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
        if j is None: us.append(a)
        else: used[j] = True
    uq = [qx[j] for j in range(len(qx)) if not used[j]]
    k = min(len(us), len(uq))
    return ';'.join(f'{wclass(a)}>{wclass(b)}' for a, b in zip(us[:k], uq[:k])), ';'.join(f'{a}>{b}' for a, b in zip(us[:k], uq[:k]))
u = pl.read_parquet('usi_keys.parquet', columns=['key', 'country', 'T', 'qn', 'sn', 'p']).filter(pl.col('key') == 'SWAP|n_swap:noise|NSAME')
r = [swaps(a, b) for a, b in zip(u['qn'].to_list(), u['sn'].to_list())]
u = u.with_columns(dir=pl.Series([x[0] for x in r]), pair=pl.Series([x[1] for x in r]))
print(u.group_by('country', 'dir').agg(n=pl.len(), tr=pl.col('T').mean(), tr_rej=pl.col('T').filter(pl.col('p') < 0.5).mean(), n_rej=(pl.col('p') < 0.5).sum()).sort('country', 'n', descending=[False, True]).head(20))
print(u.filter(pl.col('T')).group_by('country', 'pair').agg(n=pl.len()).sort('n', descending=True).head(25))
f = pl.read_parquet('../fn_add_all.parquet', columns=['q', 's', 'tier', 'key', 'qn', 'sn', 'p2g', 'p3'])
r = [swaps(a, b) for a, b in zip(f['qn'].to_list(), f['sn'].to_list())]
f = f.with_columns(dir=pl.Series([x[0] for x in r]), pair=pl.Series([x[1] for x in r]))
print(f.group_by('tier', 'dir').agg(n=pl.len()).sort('tier', 'n', descending=[False, True]).head(30))
print(f.filter(pl.col('tier') == 'A1').group_by('pair').agg(n=pl.len()).sort('n', descending=True).head(30))
f.write_parquet('fn_dirs.parquet'); u.write_parquet('usi_noiseswap_dirs.parquet')
