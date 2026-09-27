# Exact expected France dF of the simulated new decision vs v9y with per-pair true probabilities (Monte Carlo over pairs, exact per S1 by enumeration is heavy -> MC)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
V = f'{SCRATCH}/frfix3/verify_recall/'
NS = 259452
old = pl.read_parquet(V + 'fr_v9y.parquet'); new = pl.read_parquet(V + 'fr_sim_new.parquet')
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
sc = pl.read_parquet(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet', columns=['q', 's', 'p2'])
common = old.join(new, on=['s', 'q']).with_columns(kind=pl.lit('c'))
A = new.join(old, on=['s', 'q'], how='anti').join(add.select('q', 's', 'grp'), on=['q', 's'], how='left').with_columns(kind=pl.lit('a'))
Rm = old.join(new, on=['s', 'q'], how='anti').join(sc, on=['q', 's'], how='left').with_columns(kind=pl.lit('r'))
print('added', A.height, 'removed', Rm.height, 'removed p2', Rm['p2'].describe().rows())
touched = pl.concat([A.select('s'), Rm.select('s')]).unique()
com = common.join(touched, on='s').group_by('s').agg(c=pl.len())
def run(rates, r_rm, n_mc=400, seed=1):
    a = A.with_columns(t=pl.col('grp').replace_strict(rates, default=0.9, return_dtype=pl.Float64))
    rr = Rm.with_columns(t=pl.lit(r_rm) if r_rm is not None else pl.col('p2').cast(pl.Float64))
    ev = pl.concat([a.select('s', 'kind', 't'), rr.select('s', 'kind', 't')])
    ev = ev.join(com, on='s', how='left').with_columns(pl.col('c').fill_null(0))
    s_idx = ev['s'].rank('dense').cast(pl.Int64).to_numpy() - 1
    nS = s_idx.max() + 1
    kind = ev['kind'].to_numpy(); t = ev['t'].to_numpy(); c = com.join(ev.select('s').unique(), on='s', how='right').sort('s')
    cS = np.zeros(nS); 
    m = ev.select('s', 'c').unique().with_columns(i=pl.col('s').rank('dense').cast(pl.Int64) - 1)
    cS[m['i'].to_numpy()] = m['c'].to_numpy()
    isA = kind == 'a'; isR = kind == 'r'
    rng = np.random.default_rng(seed); tot = 0.0
    for _ in range(n_mc):
        tr = rng.random(len(t)) < t
        ntrue_extra = np.bincount(s_idx, weights=tr, minlength=nS)
        a_true = np.bincount(s_idx, weights=tr & isA, minlength=nS); a_n = np.bincount(s_idx, weights=isA, minlength=nS)
        r_true = np.bincount(s_idx, weights=tr & isR, minlength=nS); r_n = np.bincount(s_idx, weights=isR, minlength=nS)
        ntrue = cS + ntrue_extra
        def F(tp, npred):
            out = np.where((npred == 0) & (ntrue == 0), 1.0, 0.0)
            ok = tp > 0
            P = np.where(ok, tp / np.maximum(npred, 1), 0); R = np.where(ok, tp / np.maximum(ntrue, 1), 0)
            f = np.where(ok, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0)
            return np.where((npred == 0) & (ntrue == 0), 1.0, f)
        f_old = F(cS + r_true, cS + r_n); f_new = F(cS + a_true, cS + a_n)
        tot += (f_new - f_old).sum()
    return tot / n_mc / NS
EST = {'a_twin': 0.9, 'b_A1': 0.91, 'b_A2': 0.96, 'b_A3': 0.92, 'b_A4': 0.92, 'c_ACRONYM||NSAME': 0.95, 'c_ACRONYM+LEGAL_DROP||NSAME': 0.95, 'c_DOMAIN||NSAME': 0.93,
       'c_LEGAL_DOT||NSAME': 0.95, 'c_ACRONYM||NMISS': 0.85, 'c_ACRONYM+LEGAL_DROP||NMISS': 0.85, 'c_DOMAIN||NMISS': 0.85, 'c_LEGAL_DOT||NMISS': 0.9}
PESS = {'a_twin': 0.5, 'b_A1': 0.80, 'b_A2': 0.88, 'b_A3': 0.80, 'b_A4': 0.85, 'c_ACRONYM||NSAME': 0.88, 'c_ACRONYM+LEGAL_DROP||NSAME': 0.88, 'c_DOMAIN||NSAME': 0.75,
       'c_LEGAL_DOT||NSAME': 0.9, 'c_ACRONYM||NMISS': 0.75, 'c_ACRONYM+LEGAL_DROP||NMISS': 0.75, 'c_DOMAIN||NMISS': 0.65, 'c_LEGAL_DOT||NMISS': 0.8}
print('proposal rates, removed at their p2:', round(run(EST, None), 5), '| removed at 0.9:', round(run(EST, 0.9), 5))
print('pessimistic rates, removed at their p2:', round(run(PESS, None), 5))
for u in (0.6, 0.7, 0.72, 0.75, 0.8, 0.9):
    print(f'  uniform {u}:', round(run({k: u for k in EST}, None), 5))
