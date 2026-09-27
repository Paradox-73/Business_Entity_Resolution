import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
OUT = 'C:/ber_scratch/final2/fr-strong-veto/'
c = pl.read_parquet(OUT+'fr_cand_lineage.parquet')
FRN = r"developpement|groupe|associes|fils|\bcie\b|france|services|partenaires|\bet\b|&|\+|\bets\b|freres|holding|international|conseil"
added_noise = pl.col('qq').str.contains(FRN) & ~pl.col('ss').str.contains(FRN)
c = c.with_columns(frnoise=added_noise)
print(c.group_by('k','frnoise').agg(n=pl.len(), t03=((pl.col('ob')<0.3)&(pl.col('gb')<0.3)).sum(), recov=pl.col('in_recall_add_set(v9z?)').sum(), v7ens=pl.col('in_v7ens').sum()).sort('k','frnoise'))
r = c.filter(~pl.col('frnoise') & pl.col('k').is_in(['legal','other','same_name']) & (pl.col('ob')<0.3) & (pl.col('gb')<0.3) & (pl.col('n_v')>1))
print('clean subset n', r.height, 'recov', r['in_recall_add_set(v9z?)'].sum(), 'lin', r.group_by('lin').len().rows())
print(r.group_by('nm').len().sort('len', descending=True).head(12))
print(r.select(low=pl.col('low').mean(), up=pl.col('num').str.contains('up').mean(), amiss=pl.col('amiss').mean(), tw=(pl.col('ntw')>1).mean(), alt=pl.col('alt').mean()))
print(r.sample(min(25, r.height), seed=7).select('sn','qn','sa','qa',pl.col('ob').round(3),pl.col('gb').round(3),'nm','num'))
r.write_parquet(OUT+'clean_subset.parquet')
