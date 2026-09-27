import sys, time
import polars as pl
from multiprocessing import Pool
sys.path.insert(0, '.')
from ops import ops

def work(rows):
    return [sorted(ops(qn, qa, sn, sa, c)) for qn, qa, sn, sa, c in rows]

if __name__ == '__main__':
    t = time.time()
    d = pl.read_parquet('base.parquet')
    n = int(sys.argv[1]) if len(sys.argv) > 1 else d.height
    if n < d.height:
        d = d.sample(n, seed=5)
    rows = list(zip(d['qn'].to_list(), d['qa'].to_list(), d['sn'].to_list(), d['sa'].to_list(), d['country'].to_list()))
    ch = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
    with Pool(9) as p:
        res = [x for r in p.map(work, ch) for x in r]
    d = d.with_columns(ops=pl.Series(res, dtype=pl.List(pl.Utf8)))
    d.select('q', 'grp', 'country', 'p', 'ops').write_parquet('ops_all.parquet' if n >= 1087895 else f'ops_{n}.parquet')
    print('rows', d.height, 'sec', round(time.time() - t, 1))
