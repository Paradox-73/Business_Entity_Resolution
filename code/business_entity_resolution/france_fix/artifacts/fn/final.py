# Build the France missed-true add set (v9b-rejected best candidates with true-like census signatures) and the scores file.
import sys, re
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
sys.path.insert(0, 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/census')
import polars as pl
from ops import words, undot_legal, LEGAL
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(48); pl.Config.set_float_precision(4)
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_work.parquet').join(pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'p9best', 's9best']), on=['q', 's'])
r = d.filter(~pl.col('acc9'))
nsame = (pl.col('num2') == 'NSAME') & (pl.col('sm') == 'st')
nmiss = (pl.col('num2') == 'NMISS') & (pl.col('sm') == 'na') & (pl.col('stw') >= 0.99)
both_hi = (pl.col('p2g') >= 0.5) & (pl.col('p3') >= 0.5)
tiers = {
    'A1 noise-word swap, same number+street, not both models >= 0.5': nsame & pl.col('key').is_in(['SWAP|n_swap:noise|NSAME', 'SWAP|n_swap:desc>noise|NSAME']) & ~both_hi,
    'A2 noise-word change, house number absent, street words same': nmiss & pl.col('key').is_in([
        'SWAP|n_swap:noise|NMISS', 'SWAP|n_swap:desc>noise|NMISS', 'ADD|n_add:noise|NMISS', 'ADD+SWAP|n_add:noise n_swap:noise|NMISS',
        'ADD+SWAP|n_add:noise n_swap:desc>noise|NMISS']),
    'A3 single legal/typo change, house number absent, street words same': nmiss & pl.col('key').is_in(['LEGAL_DROP||NMISS', 'LEGAL_ADD||NMISS', 'TYPO||NMISS']),
    'A4 typo only, same number+street, transformer >= 0.5': nsame & (pl.col('key') == 'TYPO||NSAME') & (pl.col('p3') >= 0.5),
    'B1 same name, house number absent, street words same': nmiss & (pl.col('key') == '||NMISS'),
}
parts = []
for name, cond in tiers.items():
    x = r.filter(cond).with_columns(tier=pl.lit(name[:2]), tier_name=pl.lit(name))
    parts.append(x)
A = pl.concat(parts).unique(['q', 's'], keep='first')
# lowercase test per tier (name-change tiers: true ~0.1%, France decoys ~4.4%)
print(A.group_by('tier').agg(n=pl.len(), lower=pl.col('lower').mean(), nlow=pl.col('lower').sum(), g=pl.col('p2g').median(), t=pl.col('p3').median(),
      p2_2hi=(pl.col('p2_2') > 0.3).mean(), best_other_hi=((pl.col('p9best') > 0.9) & (pl.col('s9best') != pl.col('s'))).sum()).sort('tier'))
# references
ref_dec = r.filter(nsame & pl.col('key').is_in(['SWAP|n_swap:desc|NSAME', 'SWAP|n_swap:other|NSAME']))['lower'].mean()
ref_dec_nm = r.filter(nmiss & pl.col('key').is_in(['SWAP|n_swap:desc|NMISS', 'LEGAL_CHANGE||NMISS', 'LEGAL_ADD+SWAP|n_swap:other|NMISS', 'LEGAL_CHANGE+SWAP|n_swap:other|NMISS']))['lower'].mean()
print('France decoy lowercase reference: same number', round(ref_dec, 4), '| number absent', round(ref_dec_nm, 4))
# overlap with census new-swap adds
cn = pl.read_parquet('E:/Projects/Amazon ML Challenge/work/frfix2/census_add_new_swap.parquet', columns=['q', 's'])
print('overlap with census_add_new_swap:', A.join(cn, on=['q', 's']).height, 'of', cn.height)
A.select('q', 's', 'tier', 'tier_name', 'key', 'p2g', 'p3', 'p9', 'lower', 'qn', 'sn', 'qa', 'sa').write_parquet(OUT + 'fn_add_all.parquet')
for t in sorted(A['tier'].unique().to_list()):
    print(t); print(A.filter(pl.col('tier') == t).sample(6, seed=7).select('qn', 'sn', 'qa', 'sa', 'p2g', 'p3'))
