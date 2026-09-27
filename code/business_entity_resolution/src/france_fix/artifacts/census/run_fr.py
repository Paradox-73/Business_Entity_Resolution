import sys, time
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from multiprocessing import Pool
sys.path.insert(0, '.')
from ops import ops
from groups import signature

def work(rows):
    out = []
    for qn, qa, sn, sa in rows:
        o = ops(qn, qa, sn, sa, 'France')
        nc, num = signature(o)
        out.append((sorted(o), nc, num))
    return out

if __name__ == '__main__':
    t = time.time()
    d = pl.read_parquet(f'{SCRATCH}/france/fr_top.parquet', columns=['q', 's', 'p2', 'p2g', 'qn', 'qa', 'sn', 'sa', 'acc'])
    rows = list(zip(d['qn'].to_list(), d['qa'].to_list(), d['sn'].to_list(), d['sa'].to_list()))
    ch = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
    with Pool(9) as p:
        res = [x for r in p.map(work, ch) for x in r]
    d = d.select('q', 's', 'p2', 'p2g', 'acc').with_columns(
        ops=pl.Series([r[0] for r in res], dtype=pl.List(pl.Utf8)),
        nc=pl.Series([r[1] for r in res], dtype=pl.List(pl.Utf8)), num=pl.Series([r[2] for r in res]))
    d.write_parquet('fr_ops.parquet')
    print('rows', d.height, 'sec', round(time.time() - t, 1))
