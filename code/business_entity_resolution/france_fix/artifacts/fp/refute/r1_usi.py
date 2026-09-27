"""US/India label check of the veto signatures: LEGAL_ADD with number UP 1-20, and word change + UP."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
F = 'C:/ber_scratch/frfix2/fp/'
u = pl.read_parquet(F + 'usi_sig.parquet')
nm = pl.col('base').str.split('+')
hasLA = nm.list.contains('LEGAL_ADD')
hasD = nm.list.eval(pl.element().is_in(['ADD', 'SWAP', 'TYPO', 'LEGAL_CHANGE'])).list.any()
hasDrop = nm.list.eval(pl.element().is_in(['DROP', 'DUP'])).list.any()
UP = pl.col('numc') == 'UP'
u = u.with_columns(tA=UP & hasLA, tB=UP & ~hasLA & hasD & ~hasDrop, T=pl.col('grp') == 'T', acc=pl.col('p') >= 0.5)
def show(mask, name):
    x = u.filter(mask)
    print('==', name)
    print(x.group_by('country').agg(n=pl.len(), nT=pl.col('T').sum(), tr=pl.col('T').mean(), n_acc=pl.col('acc').sum(),
          tr_acc=pl.col('T').filter(pl.col('acc')).mean(), lowT=pl.col('low').filter(pl.col('T')).mean(),
          lowF=pl.col('low').filter(~pl.col('T')).mean(), nlowT=pl.col('low').filter(pl.col('T')).sum()).sort('country'))
show(pl.col('tA'), 'tier A signature (LEGAL_ADD + UP 1-20)')
show(pl.col('tB'), 'tier B signature')
show(pl.col('base') == 'LEGAL_ADD', 'LEGAL_ADD any number')
show((pl.col('base') == 'LEGAL_ADD') & (pl.col('numc') == 'NSAME'), 'LEGAL_ADD|NSAME')
show((pl.col('base') == 'LEGAL_ADD') & UP, 'LEGAL_ADD|UP')
show(UP, 'any UP')
show(UP & (pl.col('nchg') == 0), 'UP no name change')
show((pl.col('nchg') == 0) & (pl.col('numc') == 'NSAME'), 'no name change NSAME')
print('== true-record numc distribution by country (share)')
print(u.filter(pl.col('T')).group_by('country', 'numc').len().with_columns(sh=pl.col('len') / pl.col('len').sum().over('country')).sort('country', 'len', descending=True).head(30))
print('== D/W numc distribution')
print(u.filter(~pl.col('T')).group_by('country', 'numc').len().with_columns(sh=pl.col('len') / pl.col('len').sum().over('country')).sort('country', 'len', descending=True).head(30))
print('== UP breakdown by base among T in US/India (top)')
print(u.filter(UP).group_by('base').agg(n=pl.len(), nT=pl.col('T').sum(), tr=pl.col('T').mean(), n_acc=pl.col('acc').sum(), tr_acc=pl.col('T').filter(pl.col('acc')).mean()).sort('n', descending=True).head(25))
