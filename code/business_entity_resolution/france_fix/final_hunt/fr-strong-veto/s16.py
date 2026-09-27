import polars as pl
exec(open('C:/ber_scratch/final2/fr-strong-veto/s15_clean.py', encoding='utf8').read().split("r = feats(r)")[0])
r = feats(r); p = feats(p); ho = feats(ho)
okl = ~(pl.col('ops').str.contains('n_squash|n_domain') | pl.col('qn').str.contains('[@#]')) & pl.col('nm').str.contains('n_legal|n_swap|n_add|n_drop|n_typo|n_acronym|n_lower')
S2 = [pl.len().alias('n'), pl.col('low').filter(okl).mean().round(4).alias('low_chg'), pl.col('low').filter(okl).count().alias('n_chg'), pl.col('street').mean().round(3).alias('street'), pl.col('amiss2').mean().round(3).alias('amiss'), pl.col('up').mean().round(3).alias('up'), pl.col('legal_add').mean().round(3).alias('ladd'), (pl.col('ntw')>1).mean().round(3).alias('twin') if 'ntw' in r.columns else pl.lit(None).alias('twin')]
print('FR clean', r.select(S2).row(0)); print('FR keep', p.select(S2).row(0))
S3 = S2[:-1]
print(ho.group_by('label').agg(S3))
st = r.filter(pl.col('street'))
print(st.select(S2).row(0), 'ob', st['ob'].mean(), 'gb', st['gb'].mean(), 'recov', st['in_recall_add_set(v9z?)'].sum(), 'nv1', (st['n_v']==1).sum())
print(ho.filter(pl.col('street')).group_by('label').len())
print(p.filter(pl.col('street')).height, 'keep with street chg; sample:'); print(p.filter(pl.col('street')).sample(8, seed=1).select('sn','qn','sa','qa'))
