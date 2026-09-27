"""LB history check: for the France pair sets changed by v7m / v7j / v7i / v7g_num (vs v7ens), what the signature rulebook
predicts for their true rate: (a) US/India true rate of the same signature among rejected-like (p<.5) or accepted-like
(p>=.5) pairs, (b) France lowercase test within name-change signatures (true ~0, fake = US/India lowF of the signature)."""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
T = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/'
u = pl.read_parquet(F + 'usi_sig.parquet')
TT = pl.col('grp') == 'T'
ref = u.group_by('bsig').agg(tr_all=TT.mean(), tr_rej=TT.filter(pl.col('p') < 0.5).mean(), tr_acc=TT.filter(pl.col('p') >= 0.5).mean(),
                             n_rej=(pl.col('p') < 0.5).sum(), lowT=pl.col('low').filter(TT).mean(), lowF=pl.col('low').filter(~TT).mean())
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc', 'acc9', 'numc', 'low', 'bsig', 'nchg']).join(ref, on='bsig', how='left')
top = fr.select('q', 's')
E = pl.read_parquet(T + 'france/pairs_v7ens.parquet')
sets = {'v7m': T + 'frfix/namechg/pairs_v7m.parquet', 'v7j': T + 'france/pairs_v7j.parquet',
        'v7i': T + 'france/pairs_v7i.parquet', 'v7g_num': T + 'france/pairs_v7g_num.parquet'}
LB = {'v7m': -0.0091, 'v7j': -0.0163, 'v7i': -0.0035, 'v7g_num': '-0.002..0'}


def summarize(x, mode):
    """mode add: pairs were rejected in v7ens -> use tr_rej; mode rem: pairs were accepted -> tr_acc."""
    col = 'tr_rej' if mode == 'add' else 'tr_acc'
    has = x.filter(pl.col(col).is_not_null())
    # lowercase test only on name-change signatures where US/India separates (lowF - lowT >= 0.02)
    lt = x.filter((pl.col('nchg') >= 1) & ((pl.col('lowF') - pl.col('lowT')) >= 0.02))
    if lt.height:
        s, rT, rF = lt['low'].mean(), lt['lowT'].mean(), lt['lowF'].mean()
        tl = 1 - (s - rT) / (rF - rT)
    else:
        tl = None
    return dict(n=x.height, in_top=has.height, pred_true_usi=round(has[col].mean(), 3),
                n_lowtest=lt.height, low=round(lt['low'].mean(), 4) if lt.height else None,
                pred_true_low=round(tl, 3) if tl is not None else None)


for nm, p in sets.items():
    V = pl.read_parquet(p)
    add = V.join(E, on=['s', 'q'], how='anti')
    rem = E.join(V, on=['s', 'q'], how='anti')
    xa = add.join(fr, on=['q', 's'], how='inner')
    xr = rem.join(fr, on=['q', 's'], how='inner')
    print(f'== {nm}: LB France-part delta {LB[nm]}; added {add.height} (in best-cand table {xa.height}), removed {rem.height} ({xr.height})')
    if xa.height:
        print('   added  :', summarize(xa, 'add'))
        print('   added top signatures:', xa.group_by('bsig').agg(n=pl.len(), tr_rej=pl.col('tr_rej').first(), low=pl.col('low').mean()).sort('n', descending=True).head(8).to_dicts())
    if xr.height:
        print('   removed:', summarize(xr, 'rem'))
        print('   removed top signatures:', xr.group_by('bsig').agg(n=pl.len(), tr_acc=pl.col('tr_acc').first(), low=pl.col('low').mean()).sort('n', descending=True).head(8).to_dicts())
