# expected France F0.5 change with sub-group true rates (per touched S1, exact enumeration over its added pairs)
import polars as pl, itertools, sys
sys.path.insert(0, 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census')
NS1 = 259452
g = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/sim_added_c.parquet')
dirs = pl.read_parquet('fn_dirs.parquet', columns=['q', 's', 'dir']).join(pl.read_parquet('fn_nw.parquet', columns=['q', 's', 'nw']), on=['q', 's'])
A = pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/fn_add_all.parquet', columns=['q', 's', 'tier', 'qn', 'sn'])
A = A.join(dirs, on=['q', 's'], how='left')
X = ['holding', 'international', 'distribution', 'participations']
A = A.with_columns(sub=pl.when((pl.col('tier') == 'A1') & pl.col('nw').is_in(X)).then(pl.lit('A1x'))
                   .when((pl.col('tier') == 'A1') & (pl.col('dir').str.contains(';') | pl.col('dir').str.starts_with('noise>other'))).then(pl.lit('A1q'))
                   .when((pl.col('tier') == 'A4') & pl.col('qn').str.to_lowercase().str.contains('fetes') & pl.col('sn').str.to_lowercase().str.contains(r'\bets\b')).then(pl.lit('A4q'))
                   .otherwise(pl.col('tier')))
print(A['sub'].value_counts().sort('sub'))
g = g.join(A.select('q', 's', 'sub'), on=['q', 's'], how='left')
def F(tp, fn, fp):
    if tp == 0 and fn == 0 and fp == 0: return 1.0
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
def run(T, drop=()):
    x = g.filter(~pl.col('sub').is_in(list(drop)))
    tot = 0.0
    for (s,), grp in x.group_by(['s']):
        k = grp['c'][0]; ts = [T[u] for u in grp['sub'].to_list()]
        for outc in itertools.product([0, 1], repeat=len(ts)):
            pr = 1.0
            for o, t in zip(outc, ts): pr *= t if o else 1 - t
            xt = sum(outc); m = len(ts)
            tot += pr * (F(k + xt, 0, m - xt) - F(k, xt, 0))
    return tot / NS1, x.height
scen = {'central': {'A1': 0.92, 'A1q': 0.5, 'A1x': 0.05, 'A2': 0.88, 'A3': 0.85, 'A4': 0.90, 'A4q': 0.1},
        'optimistic (analyst)': {'A1': 0.95, 'A1q': 0.95, 'A1x': 0.95, 'A2': 0.97, 'A3': 0.95, 'A4': 0.92, 'A4q': 0.92},
        'pessimistic (x0.82, like v9b noise LB-fit)': {'A1': 0.75, 'A1q': 0.3, 'A1x': 0.05, 'A2': 0.72, 'A3': 0.70, 'A4': 0.74, 'A4q': 0.1}}
for nm, T in scen.items():
    full, n1 = run(T); core, n2 = run(T, drop=('A1q', 'A1x', 'A4q'))
    mean_t = sum(T[u] for u in g['sub'].to_list()) / g.height
    print(f'{nm:45s} mean t {mean_t:.3f} | full {n1} pairs: France {full:+.5f} LB {0.15*full:+.6f} | without A1q/A1x/A4q {n2} pairs: France {core:+.5f} LB {0.15*core:+.6f}')
A.select('q', 's', 'tier', 'sub').write_parquet('fn_sub.parquet')
