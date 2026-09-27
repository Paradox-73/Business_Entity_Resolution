import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
OUT = 'C:/ber_scratch/final2/fr-strong-veto/'
c = pl.read_parquet(OUT+'fr_cand_lineage.parquet')
c = c.with_columns(lin=pl.when(pl.col('in_recall_add_set(v9z?)')).then(pl.lit('3_v9z_recov')).when(pl.col('in_v7ens')&pl.col('in_v7g')).then(pl.lit('2a_v7ens_keptv7g')).when(pl.col('in_v7ens')).then(pl.lit('2b_v7ens_remv7g')).when(pl.col('in_v7m')).then(pl.lit('1_v7m_restored')).otherwise(pl.lit('4_v9y_new')))
n = lambda e: e.fill_null('').str.normalize('NFKD').str.replace_all(r'\p{M}','').str.to_lowercase()
c = c.with_columns(qq=n(pl.col('qn')), ss=n(pl.col('sn')))
tok = lambda t: (pl.col('qq').str.contains(t, literal=True) & ~pl.col('ss').str.contains(t, literal=True)).mean().round(3).alias(t)
aggs = [pl.len().alias('n'), ((pl.col('ob')<0.2)&(pl.col('gb')<0.2)).sum().alias('t02'), pl.col('ob').mean().round(3).alias('ob'), pl.col('gb').mean().round(3).alias('gb'), pl.col('og').mean().round(3).alias('og'), pl.col('osm').mean().round(3).alias('osm'),
        pl.col('k').eq('drop/swap+noise').mean().round(3).alias('dropswap'), pl.col('k').eq('legal').mean().round(3).alias('legal'), pl.col('amiss').mean().round(3).alias('amiss'), (pl.col('num')=='a_num_missing').mean().round(3).alias('nmiss'),
        pl.col('n_v').mean().round(2).alias('nv'), (pl.col('n_v')==1).mean().round(3).alias('nv1'), (pl.col('ntw')>1).mean().round(3).alias('tw')] + [tok(t) for t in ['developpement','groupe','associes','& fils','cie']]
print(c.group_by('lin').agg(aggs).sort('lin'))
c.write_parquet(OUT+'fr_cand_lineage.parquet')
for l in ['3_v9z_recov','1_v7m_restored']:
    print(l); print(c.filter(pl.col('lin')==l).sample(12, seed=5).select('sn','qn','sa','qa',pl.col('ob').round(3),pl.col('gb').round(3),'nm'))
