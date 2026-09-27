"""France number-keeping test: for each name-change type (base signature name part, optionally x word class), the share
of France best-candidate pairs that keep the house number. Decoys keep it ~1-3% (pure decoy words), true pairs ~80-95%.
Expected decoys among same-number pairs ~= r * (number moved up pairs)."""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
F = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fp/'
fr = pl.read_parquet(F + 'fr_sig.parquet', columns=['q', 's', 'acc9', 'veto', 'numc', 'low', 'base', 'nchg', 'p2'])
fr = fr.join(pl.read_parquet(F + 'fr_cls.parquet'), on=['q', 's']).join(pl.read_parquet(F + 'fr_fk.parquet'), on=['q', 's'])
UPS = pl.col('numc').is_in(['UP', 'UPbig'])
R = 0.02   # decoys: same-number / moved-up


def tab(keys):
    g = fr.group_by(keys).agg(
        n=pl.len(), n_same=(pl.col('numc') == 'NSAME').sum(), n_up=UPS.sum(),
        acc_same=((pl.col('numc') == 'NSAME') & pl.col('acc9')).sum(),
        acc_all=pl.col('acc9').sum(),
        low_acc_same=pl.col('low').filter((pl.col('numc') == 'NSAME') & pl.col('acc9')).mean(),
        low_up=pl.col('low').filter(UPS).mean())
    g = g.with_columns(keep=pl.col('n_same') / pl.col('n'), exp_dec_same=R * pl.col('n_up'))
    return g.with_columns(dec_share_same=(pl.col('exp_dec_same') / pl.col('n_same')).clip(0, 1))


g = tab(['base'])
print('== by name-change type (all France best-candidate pairs); exp_dec_same = 0.02 * n_up')
print(g.filter(pl.col('n') >= 300).sort('n', descending=True).head(50))
g2 = tab(['base', 'cls'])
print('== by type x word class: groups with many accepted same-number pairs and high decoy share')
print(g2.filter((pl.col('acc_same') >= 100)).sort('dec_share_same', descending=True).head(40))
g3 = tab(['base', 'fk'])
print('== by type x name-overlap kind')
print(g3.filter((pl.col('acc_same') >= 100)).sort('dec_share_same', descending=True).head(30))
