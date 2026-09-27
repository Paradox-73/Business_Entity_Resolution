# variant fpfn: v9b scores + fp veto (1,145 -> p2 0) + fn clean adds minus lowercase and sibling-ambiguous (-> p2 0.95)
import polars as pl
W = 'E:/Projects/Amazon ML Challenge/work/'
B = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/build/'
F = pl.read_parquet(B + 'fn_clean_flags.parquet')
add = F.filter(~pl.col('lower') & (pl.col('namb') == 0)).select('q', 's')
print('fn clean', F.height, 'lowercase', F['lower'].sum(), 'ambiguous', (F['namb'] > 0).sum(), '-> add', add.height)
print(F.filter(~pl.col('lower') & (pl.col('namb') == 0)).group_by('tier').len().sort('tier'))
add.write_parquet(W + 'frfix2/fn_add_set_final.parquet')
veto = pl.read_parquet(W + 'frfix2/fp_veto_set.parquet', columns=['q', 's'])
assert add.join(veto, on=['q', 's']).height == 0 and add.join(veto, on='q').height == 0
sc = pl.read_parquet(W + 'frfix/test_scores_frmin_descveto_gnadd.parquet')
print(sc.schema, sc.height)
n0 = sc.height
sc = (sc.join(veto.with_columns(v=pl.lit(True)), on=['q', 's'], how='left')
        .join(add.with_columns(a=pl.lit(True)), on=['q', 's'], how='left')
        .with_columns(p2=pl.when(pl.col('v').fill_null(False)).then(pl.lit(0.0))
                        .when(pl.col('a').fill_null(False)).then(pl.lit(0.95)).otherwise(pl.col('p2')).cast(pl.Float32))
        .drop('v', 'a'))
assert sc.height == n0
print('veto rows hit', sc.join(veto, on=['q', 's']).height, 'add rows hit', sc.join(add, on=['q', 's']).height)
sc.write_parquet(W + 'frfix2/test_scores_v9b_fpfn.parquet')
print('written')
