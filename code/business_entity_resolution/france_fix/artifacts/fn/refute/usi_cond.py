# US/India: true rate of the tier signatures, conditional on the model rejecting / transformer lowering (the analog of v9b-rejected France pairs)
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50); pl.Config.set_float_precision(4)
C = 'C:/ber_scratch/frfix2/census/'
u = pl.read_parquet(C + 'sig_all.parquet', columns=['q', 'grp', 'country', 'p', 'num', 'nc']).join(
    pl.read_parquet(C + 'ops_all.parquet', columns=['q', 'ops']), on='q')
u = u.with_columns(lower=pl.col('ops').list.contains('n_lower'), num2=pl.col('num').str.replace(r'^(UP|DOWN).*', '$1'),
                   bnc=pl.col('nc').list.filter(pl.element() != 'LOWER').list.join('+'),
                   wcls=pl.col('ops').list.eval(pl.element().filter(pl.element().str.contains(r'^n_(swap|add|drop):'))).list.unique().list.sort().list.join(' '))
u = u.with_columns(key=pl.col('bnc') + '|' + pl.col('wcls') + '|' + pl.col('num2'), T=pl.col('grp') == 'T')
b = pl.read_parquet('C:/ber_scratch/france/usi_top.parquet', columns=['q', 's', 'p2g', 'p3', 'label', 'qn', 'sn'])
u = u.join(b, on='q', how='left')
print('label agrees with grp T:', (u['label'] == u['T']).mean(), u.height)
keys = {'A1': ['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME'],
        'A2': ['SWAP|n_swap:noise|NMISS', 'SWAP|n_swap:desc>noise|NMISS', 'ADD|n_add:noise|NMISS', 'ADD+SWAP|n_add:noise n_swap:noise|NMISS', 'ADD+SWAP|n_add:noise n_swap:desc>noise|NMISS'],
        'A3': ['LEGAL_DROP||NMISS', 'LEGAL_ADD||NMISS', 'TYPO||NMISS'],
        'A4': ['TYPO||NSAME'], 'B1': ['||NMISS']}
rows = []
for t, ks in keys.items():
    x = u.filter(pl.col('key').is_in(ks))
    for cname, cond in [('all', pl.lit(True)), ('p<0.5 (rejected)', pl.col('p') < 0.5), ('p<0.5 & p3<0.5', (pl.col('p') < 0.5) & (pl.col('p3') < 0.5)),
                        ('p3<0.2', pl.col('p3') < 0.2), ('p<0.5 & p3<0.2', (pl.col('p') < 0.5) & (pl.col('p3') < 0.2)),
                        ('p<0.5 & not(both>=0.5)', (pl.col('p') < 0.5) & ~((pl.col('p2g') >= 0.5) & (pl.col('p3') >= 0.5))),
                        ('p<0.5 & p3>=0.5', (pl.col('p') < 0.5) & (pl.col('p3') >= 0.5))]:
        y = x.filter(cond)
        for c in ['US', 'India']:
            z = y.filter(pl.col('country') == c)
            if z.height:
                rows.append(dict(tier=t, cond=cname, country=c, n=z.height, tr=z['T'].mean(), W=(z['grp'] == 'W').mean(), D=(z['grp'] == 'D').mean(),
                                 lowT=z.filter(pl.col('T'))['lower'].mean() if z['T'].sum() else None,
                                 lowF=z.filter(~pl.col('T'))['lower'].mean() if (~z['T']).sum() else None,
                                 nT=int(z['T'].sum()), nF=int((~z['T']).sum())))
print(pl.DataFrame(rows))
u.select('q', 's', 'grp', 'country', 'p', 'p2g', 'p3', 'key', 'lower', 'T', 'qn', 'sn', 'ops').write_parquet('usi_keys.parquet')
