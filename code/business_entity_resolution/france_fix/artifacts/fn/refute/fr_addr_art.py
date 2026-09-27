# second artifact: address ops that separate true/false in US/India (a_chg:place, a_chg:street, a_chg:unit, a_extra_numbers ...), measured on France groups
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(40); pl.Config.set_float_precision(4)
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_work.parquet', columns=['q', 's', 'p2g', 'p3', 'acc9', 'key', 'num2', 'sm', 'stw'])
d = d.join(pl.read_parquet('fr_low_own.parquet'), on=['q', 's'])
d = d.join(pl.read_parquet('C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census/fr_ops.parquet', columns=['q', 's', 'ops']), on=['q', 's'], how='left')
A = pl.read_parquet(OUT + 'fn_add_all.parquet', columns=['q', 's', 'tier'])
d = d.join(A, on=['q', 's'], how='left')
A1k = ['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME']
DECk = ['SWAP|n_swap:desc|NSAME', 'SWAP|n_swap:other|NSAME']
st = (pl.col('sm') == 'st')
grp = (pl.when(pl.col('tier').is_not_null()).then(pl.col('tier'))
       .when(st & pl.col('key').is_in(A1k) & pl.col('acc9')).then(pl.lit('acc noise-swap same addr'))
       .when(st & pl.col('key').is_in(DECk) & ~pl.col('acc9')).then(pl.lit('rej desc/other swap same addr (decoy ref)'))
       .when(st & pl.col('key').is_in(DECk) & pl.col('acc9')).then(pl.lit('acc desc/other swap same addr'))
       .when(st & (pl.col('key') == 'TYPO||NSAME') & pl.col('acc9')).then(pl.lit('acc typo same addr'))
       .when(st & (pl.col('key') == 'TYPO||NSAME') & ~pl.col('acc9') & (pl.col('p3') < 0.5)).then(pl.lit('rej typo same addr p3<0.5'))
       .when(st & pl.col('key').is_in(A1k) & pl.col('veto')).then(pl.lit('vetoed noise-swap same addr'))
       .when(st & (pl.col('key') == '||NSAME') & pl.col('acc9')).then(pl.lit('acc same name same addr'))
       .when(pl.col('veto')).then(pl.lit('vetoed (all)'))
       .otherwise(None))
d = d.with_columns(g=grp).filter(pl.col('g').is_not_null())
opsl = ['a_chg:place', 'a_chg:street', 'a_chg:unit', 'a_chg:state', 'a_extra_numbers', 'a_add:street', 'a_add:unit', 'a_drop:place', 'n_upper', 'n_acc_add', 'n_acc_strip', 'n_dblspace', 'a_upper', 'a_lower', 'n_hyphen', 'n_bracket']
aggs = [pl.len().alias('n'), pl.col('low').mean().alias('LOW')] + [pl.col('ops').list.contains(o).mean().alias(o) for o in opsl]
print(d.group_by('g').agg(aggs).sort('g'))
# same on US/India labels for the A1 key
u = pl.read_parquet('usi_keys.parquet', columns=['key', 'country', 'T', 'ops', 'lower']).filter(pl.col('key').is_in(A1k + ['TYPO||NSAME']))
aggs[1] = pl.col('lower').mean().alias('LOW')
print(u.group_by('country', 'T').agg(aggs).sort('country', 'T'))
