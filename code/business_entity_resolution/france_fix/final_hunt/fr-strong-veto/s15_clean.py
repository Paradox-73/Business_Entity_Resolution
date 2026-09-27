import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60)
OUT = 'C:/ber_scratch/final2/fr-strong-veto/'
r = pl.read_parquet(OUT+'clean_subset.parquet')
p = pl.read_parquet(OUT+'fr_prof.parquet').filter(pl.col('grp')=='keep_cc')
ho = pl.read_parquet(OUT+'ho_prof.parquet').filter((pl.col('w')=='ab')&(pl.col('ob')<0.3)&(pl.col('gb')<0.3)&pl.col('k').is_in(['legal','other','same_name']))
def feats(d):
    inval = pl.col('ops').str.contains('n_lower|n_squash|n_domain') | pl.col('qn').str.contains('[@#]')
    chg = pl.col('nm').str.contains('n_legal|n_swap|n_add|n_drop|n_typo|n_acronym')
    return d.with_columns(lowv=pl.when(~inval & chg).then(pl.col('low')), street=pl.col('ops').str.contains('a_chg:street'),
                          amiss2=pl.col('ops').str.contains('a_missing'), up=pl.col('ops').str.contains('a_num_up'), dn=pl.col('ops').str.contains('a_num_down'),
                          legal_add=pl.col('nm').str.contains('n_legal_add'), legal_drop=pl.col('nm').str.contains('n_legal_drop'), same=(pl.col('nm')==''))
S = [pl.len().alias('n'), pl.col('lowv').mean().round(4).alias('low_valid'), pl.col('lowv').count().alias('n_lowvalid'), pl.col('street').mean().round(3).alias('street_chg'), pl.col('amiss2').mean().round(3).alias('addr_missing'),
     pl.col('up').mean().round(3).alias('num_up'), pl.col('dn').mean().round(3).alias('num_dn'), pl.col('legal_add').mean().round(3).alias('legal_add'), pl.col('legal_drop').mean().round(3).alias('legal_drop'), pl.col('same').mean().round(3).alias('same_name')]
r = feats(r); p = feats(p); ho = feats(ho)
print('FR clean subset'); print(r.select(S))
print('FR keep_cc (confident, mostly true)'); print(p.select(S))
print('FR keep_cc restricted to legal/other/same_name classes')
def cls(e):
    return (pl.when(e.str.contains("n_swap:noise|n_swap:desc") | (e.str.contains("n_drop") & e.str.contains("n_add:noise"))).then(pl.lit("drop/swap+noise"))
            .when(e.str.contains("n_legal")).then(pl.lit("legal")).when(e=="").then(pl.lit("same_name")).when(e.str.contains("n_add:noise") & ~e.str.contains("n_swap|n_drop")).then(pl.lit("add_noise_only")).otherwise(pl.lit("other")))
print(p.with_columns(k=cls(pl.col('nm'))).filter(pl.col('k').is_in(['legal','other','same_name'])).select(S))
print('US/IN analog same classes, by label'); print(ho.group_by('label').agg(S))
# sub-slices of the clean subset
for nm_, f in [('street_chg', pl.col('street')), ('no_street_chg', ~pl.col('street')), ('legal_add & !street', pl.col('legal_add') & ~pl.col('street')), ('same_name', pl.col('same'))]:
    x = r.filter(f); print(nm_, x.select(S).row(0))
