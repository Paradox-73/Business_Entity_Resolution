import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
OUT = f'{SCRATCH}/final2/fr-strong-veto/'
d = pl.read_parquet(OUT+'clean_subset_rep.parquet')
lin = pl.read_parquet(OUT+'fr_cand_lineage.parquet').select('s','q','in_recall_add_set(v9z?)','lin')
d = d.drop([c for c in ['in_recall_add_set(v9z?)','lin'] if c in d.columns]).join(lin, on=['s','q'], how='left')
v = d.filter(pl.col('rep').fill_null(False) & ~pl.col('in_recall_add_set(v9z?)') & (pl.col('n_v')>1) & (pl.col('ob')<0.3) & (pl.col('gb')<0.3))
print('veto set n', v.height, 'rows', v['s'].n_unique(), 'n_v dist', v['n_v'].value_counts().sort('n_v').rows())
print(v.select(pl.col('og').mean(), pl.col('gg').mean(), pl.col('ob').mean(), pl.col('gb').mean(), pl.col('oe').mean(), (pl.col('ntw')>1).mean(), pl.col('twin_street_hit').sum(), pl.col('low').mean()))
print(v.group_by('lin').len().rows())
f = lambda P, R: 0 if P+R == 0 else 1.25*P*R/(0.25*P+R)
N = 259452
for t in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75):
    tot = 0.0
    for k in v['n_v'].to_list():
        gf = f(1.0, 1.0) - f((k-1)/k, 1.0)      # remove a false pair from a row whose other k-1 are true
        lt = f(1.0, 1.0) - f(1.0, (k-1)/k)      # remove a true pair from a fully correct row
        tot += (1-t)*gf - t*lt
    print(f"t={t}: dF_France={tot/N:+.6f}  dLB={0.15*tot/N:+.7f}")
v.select('s','q','sn','sa','qn','qa','ob','gb','oe','og','gg','n_v','ntw','twin_street_hit','nm','ops','lin').write_parquet(OUT+'fr_strong_veto_street.parquet')
print(v.select('sn','sa','qn','qa',pl.col('ob').round(3),pl.col('gb').round(3)).sample(15, seed=11))
