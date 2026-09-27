"""Per base signature (name changes without LOWER | number class): US/India lowercase share of true vs false, true rate
among accepted-like pairs (p >= 0.5), and France accepted (v9b, not vetoed) count + lowercase share -> fake estimate."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(50)
F = f'{SCRATCH}/frfix2/fp/'
u = pl.read_parquet(F + 'usi_sig.parquet')
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'p2', 'acc', 'acc9', 'veto', 'gadd', 'numc', 'low', 'base', 'nchg', 'bsig'])

T = pl.col('grp') == 'T'
ref = u.group_by('bsig').agg(
    nT=T.sum(), nF=(~T).sum(), lowT=pl.col('low').filter(T).mean(), lowF=pl.col('low').filter(~T).mean(),
    tr=T.mean(), n_acc=(pl.col('p') >= 0.5).sum(), tr_acc=T.filter(pl.col('p') >= 0.5).mean(),
    tr_acc_nolow=T.filter((pl.col('p') >= 0.5) & ~pl.col('low')).mean(),
    tr_us=T.filter(pl.col('country') == 'US').mean(), tr_in=T.filter(pl.col('country') == 'India').mean())
ref.write_parquet(F + 'ref_bsig.parquet')

fa = fr.filter(pl.col('acc9'))
g = (fa.group_by('bsig').agg(n=pl.len(), nchg=pl.col('nchg').first(), low=pl.col('low').mean(), nlow=pl.col('low').sum(),
                             p2=pl.col('p2').mean(), gadd=pl.col('gadd').sum())
       .join(fr.filter(~pl.col('acc9') & ~pl.col('veto')).group_by('bsig').agg(n_rej=pl.len(), low_rej=pl.col('low').mean()), on='bsig', how='left')
       .join(ref, on='bsig', how='left'))
# fake share from lowercase: use France reference rT=0.002, rD=0.038 (flat), only meaningful for nchg >= 1
g = g.with_columns(fake_low=((pl.col('low') - 0.002) / (0.038 - 0.002)).clip(0, 1))
g = g.with_columns(fake_n=pl.col('fake_low') * pl.col('n'))
g.write_parquet(F + 'fr_acc_bsig.parquet')
cols = ['bsig', 'n', 'low', 'n_rej', 'low_rej', 'lowT', 'lowF', 'nT', 'nF', 'tr', 'tr_acc', 'tr_us', 'tr_in', 'fake_low', 'fake_n', 'gadd']
print('== accepted France groups nchg>=1, n>=100, sorted by fake_n (lowercase-based)')
print(g.filter((pl.col('nchg') >= 1) & (pl.col('n') >= 100)).sort('fake_n', descending=True).select(cols).head(60))
print('== accepted France groups with tr_acc < 0.76 (US/India true rate among p>=.5), n>=50')
print(g.filter((pl.col('tr_acc') < 0.76) & (pl.col('n') >= 50)).sort('n', descending=True).select(cols).head(60))
