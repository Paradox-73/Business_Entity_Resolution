"""Tier C composition, lowT sources, and DOWN-mirror estimate per tier on accepted v9b pairs."""
import polars as pl
from scipy.stats import poisson, chi2
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(50)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
v = pl.read_parquet(F + 'fp_veto_set.parquet')
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'acc9', 'veto', 'gadd', 'numc', 'num', 'low', 'base', 'nchg', 'bsig'])
ref = pl.read_parquet(F + 'ref_bsig.parquet')
v = v.join(fr.select('q', 's', 'gadd', 'acc9'), on=['q', 's'], how='left')
print('veto pairs in v9b noise-word additions (gadd):', v.group_by('tier').agg(n=pl.len(), gadd=pl.col('gadd').sum(), acc9=pl.col('acc9').sum()).sort('tier').to_dicts())
c = v.filter(pl.col('tier') == 'C').join(ref.select('bsig', 'nT', 'lowT', 'lowF'), on='bsig', how='left')
print(c.group_by('bsig').agg(n=pl.len(), t=pl.col('t').first(), nT=pl.col('nT').first(), lowT=pl.col('lowT').first(), nlowT=(pl.col('lowT') * pl.col('nT')).first().round(0), lowF=pl.col('lowF').first(), gadd=pl.col('gadd').sum()).sort('n', descending=True))
print(c.select('p2', 'bsig', 'qn', 'sn').head(12))
# mirror estimate
nm = pl.col('base').str.split('+')
hasLA = nm.list.contains('LEGAL_ADD')
hasD = nm.list.eval(pl.element().is_in(['ADD', 'SWAP', 'TYPO', 'LEGAL_CHANGE'])).list.any()
hasDrop = nm.list.eval(pl.element().is_in(['DROP', 'DUP'])).list.any()
fr = fr.with_columns(A=hasLA, B=~hasLA & hasD & ~hasDrop)
acc = fr.filter(pl.col('acc9'))
for t in ['A', 'B']:
    up = acc.filter(pl.col(t) & (pl.col('numc') == 'UP'))
    dn = acc.filter(pl.col(t) & (pl.col('numc') == 'DOWN'))
    nlow_dn = dn['low'].sum()
    # accepted DOWN fakes, 95% upper bound from zero lowercase (fake lowercase ~0.03)
    print(f'tier {t}: accepted UP {up.height} (lowercase {up["low"].sum()}), accepted DOWN {dn.height} (lowercase {nlow_dn})')
    for r in (0.76, 0.88, 1.0):
        print(f'   true UP ~ {r} x true DOWN: t <= {r * dn.height / up.height:.3f}')
    # lowercase test: fake lowercase rate from France tier pairs with v7ens p2 in (0.02, 0.5]
    nb = fr.filter(pl.col(t) & (pl.col('numc') == 'UP') & ~pl.col('veto') & (pl.col('p2') > 0.02) & (pl.col('p2') <= 0.5))
    rF = nb['low'].mean()
    k = int(up['low'].sum())
    lo, hi = chi2.ppf(0.025, 2 * k) / 2, chi2.ppf(0.975, 2 * k + 2) / 2
    print(f'   lowercase: fake rate (neighbour p2 .02-.5, n={nb.height}) {rF:.4f}; observed {k} vs all-fake {up.height * rF:.1f}; '
          f't = {1 - k / (up.height * rF):.2f}, 95% range {max(0, 1 - hi / (up.height * rF)):.2f}..{max(0, 1 - lo / (up.height * rF)):.2f}')
