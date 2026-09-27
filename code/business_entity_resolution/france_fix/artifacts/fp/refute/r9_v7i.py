"""v7i added 341 tier-A-signature pairs with 0.3% lowercase (true-like). What separates them from the 846 accepted
tier-A veto pairs (3.4% lowercase)? Check transformer p3, GBDT p2g, num detail, examples."""
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(55)
T = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/'
F = T + 'frfix2/fp/'
E = pl.read_parquet(T + 'france/pairs_v7ens.parquet')
V = pl.read_parquet(T + 'france/pairs_v7i.parquet')
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'p2g', 'acc9', 'numc', 'num', 'low', 'base', 'veto', 'gadd'])
nm = pl.col('base').str.split('+')
fr = fr.filter((pl.col('numc') == 'UP') & nm.list.contains('LEGAL_ADD'))
p3 = pl.read_parquet('E:/Projects/Amazon ML Challenge/work/test_scores_blend_ab_a2.parquet', columns=['q', 's', 'p2']).rename({'p2': 'p3'})
fr = fr.join(p3, on=['q', 's'], how='left')
add = V.join(E, on=['s', 'q'], how='anti')
fr = fr.with_columns(v7i=pl.struct('q', 's').is_in(add.select(pl.struct('q', 's')).to_series()))
top = pl.read_parquet(T + 'france/fr_top.parquet', columns=['q', 's', 'qn', 'sn', 'qa', 'sa'])
g = fr.with_columns(grp=pl.when(pl.col('acc9')).then(pl.lit('v9b_acc')).when(pl.col('v7i')).then(pl.lit('v7i_add')).otherwise(pl.lit('rej')))
print(g.group_by('grp').agg(n=pl.len(), low=pl.col('low').mean(), p2=pl.col('p2').mean(), p2g=pl.col('p2g').mean(), p3=pl.col('p3').mean(), veto=pl.col('veto').sum(), gadd=pl.col('gadd').sum()).sort('grp'))
print('v9b accepted tier-A signature: lowercase by p3 bin')
print(g.filter(pl.col('grp') == 'v9b_acc').with_columns(b=pl.col('p3').cut([0.1, 0.3, 0.5, 0.8, 0.95])).group_by('b').agg(n=pl.len(), nlow=pl.col('low').sum(), low=pl.col('low').mean()).sort('b'))
print('v7i-added tier-A signature examples')
print(g.filter(pl.col('grp') == 'v7i_add').join(top, on=['q', 's']).select('p2', 'p3', 'num', 'qn', 'sn', 'qa', 'sa').head(15))
print(g.filter(pl.col('grp') == 'v7i_add').group_by('base').len().sort('len', descending=True).head(8))
print(g.filter(pl.col('grp') == 'v9b_acc').group_by('base').len().sort('len', descending=True).head(8))
