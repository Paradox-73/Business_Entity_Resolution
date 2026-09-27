"""Does the model's acceptance favour lowercase decoys? Lowercase share of FALSE pairs by score bin (US/India labels),
and of France tier-A-signature pairs by p2 bin. If accepted decoys were lowercase more often than rejected ones,
the tier-A lowercase test would overstate the fake share."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
u = pl.read_parquet(F + 'usi_sig.parquet', columns=['q', 'grp', 'country', 'p', 'base', 'numc', 'low', 'nchg'])
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'acc9', 'veto', 'gadd', 'numc', 'low', 'base', 'nchg'])
v9 = pl.read_parquet(F + 'fr_scores_v9b_lv.parquet', columns=['q', 's', 'p2']).rename({'p2': 'p9'})
fr = fr.join(v9, on=['q', 's'], how='left')
nm = pl.col('base').str.split('+')
tA = (pl.col('numc') == 'UP') & nm.list.contains('LEGAL_ADD')
tB = (pl.col('numc') == 'UP') & ~nm.list.contains('LEGAL_ADD') & nm.list.eval(pl.element().is_in(['ADD', 'SWAP', 'TYPO', 'LEGAL_CHANGE'])).list.any() & ~nm.list.eval(pl.element().is_in(['DROP', 'DUP'])).list.any()
bins = [0.02, 0.1, 0.3, 0.5, 0.8]
def binned(d, col):
    return d.with_columns(b=pl.col(col).cut(bins))
print('== US/India FALSE pairs, name change (nchg>=1) and number UP, lowercase share by p bin')
x = binned(u.filter((pl.col('grp') != 'T') & (pl.col('nchg') >= 1) & (pl.col('numc') == 'UP')), 'p')
print(x.group_by('country', 'b').agg(n=pl.len(), low=pl.col('low').mean()).sort('country', 'b'))
print('== US/India FALSE pairs, tier-A signature, by p bin')
x = binned(u.filter((pl.col('grp') != 'T') & tA), 'p')
print(x.group_by('b').agg(n=pl.len(), low=pl.col('low').mean()).sort('b'))
print('== US/India TRUE pairs, tier-A signature: lowercase and acceptance')
print(u.filter((pl.col('grp') == 'T') & tA).group_by('country').agg(n=pl.len(), low=pl.col('low').mean(), acc=(pl.col('p') >= 0.5).mean()))
print('== US/India TRUE pairs acceptance by class x numc (UP vs DOWN vs NSAME)')
cl = pl.when(nm.list.contains('LEGAL_ADD')).then(pl.lit('A')).when(pl.col('nchg') == 0).then(pl.lit('none')).otherwise(pl.lit('other'))
print(u.filter((pl.col('grp') == 'T') & pl.col('numc').is_in(['UP', 'DOWN', 'NSAME'])).with_columns(k=cl).group_by('country', 'k', 'numc')
       .agg(n=pl.len(), acc=(pl.col('p') >= 0.5).mean()).sort('country', 'k', 'numc'))
print('== France tier-A signature (not vetoed by descriptor veto), lowercase share by v7ens p2 bin')
x = binned(fr.filter(tA & ~pl.col('veto')), 'p2')
print(x.group_by('b').agg(n=pl.len(), acc9=pl.col('acc9').sum(), low=pl.col('low').mean(), nlow=pl.col('low').sum()).sort('b'))
print('== France tier-B signature by p2 bin')
x = binned(fr.filter(tB & ~pl.col('veto')), 'p2')
print(x.group_by('b').agg(n=pl.len(), acc9=pl.col('acc9').sum(), low=pl.col('low').mean(), nlow=pl.col('low').sum()).sort('b'))
print('== France all name-change (nchg>=1) UP pairs by p2 bin (not vetoed)')
x = binned(fr.filter((pl.col('nchg') >= 1) & (pl.col('numc') == 'UP') & ~pl.col('veto')), 'p2')
print(x.group_by('b').agg(n=pl.len(), acc9=pl.col('acc9').sum(), low=pl.col('low').mean()).sort('b'))
