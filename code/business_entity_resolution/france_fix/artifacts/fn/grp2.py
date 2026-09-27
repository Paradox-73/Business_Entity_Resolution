import sys, re, unicodedata
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from rapidfuzz import fuzz
from fr_restore import street, TYPES
pl.Config.set_tbl_rows(150); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(320); pl.Config.set_fmt_str_lengths(62); pl.Config.set_float_precision(3)
OUT = 'C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix2/fn/'
d = pl.read_parquet(OUT + 'fr_base.parquet', columns=['q', 's', 'qn', 'sn', 'qa', 'sa', 'p2g', 'p3', 'p9', 'acc9', 'p2_2', 'ops'])
d = d.join(pl.read_parquet(OUT + 'fr_key2.parquet', columns=['q', 's', 'key', 'bnc', 'wcls', 'num2', 'lower', 'sm']), on=['q', 's'])
d = d.join(pl.read_parquet(OUT + 'fr_street.parquet', columns=['q', 's', 'sm2']), on=['q', 's'])
d = d.with_columns(sfx=pl.col('ops').list.contains('a_num_suffix') | pl.col('ops').list.contains('a_num_dot'))
def norm(x):
    return unicodedata.normalize('NFKD', x or '').encode('ascii', 'ignore').decode().lower()
def stw(qa, sa):
    """share of the S1 street words (after the house number, street types dropped) found (fuzzy) in the record address"""
    n, st = street(sa)
    ws = [w for w in st.split() if len(w) > 2]
    if not ws:
        return -1.0
    qw = re.findall(r'[a-z]+', norm(qa))
    return sum(1 for w in ws if any(fuzz.ratio(w, v) >= 80 for v in qw)) / len(ws)
m = d['num2'] == 'NMISS'
sub = d.filter(m)
sub = sub.with_columns(stw=pl.Series([stw(a, b) for a, b in zip(sub['qa'].to_list(), sub['sa'].to_list())]))
d = d.join(sub.select('q', 's', 'stw'), on=['q', 's'], how='left')
d.drop('ops').write_parquet(OUT + 'fr_work.parquet')
# references for the suffix/dot artifact (same street only)
st = d.filter(pl.col('sm') == 'st')
print('sfx rate: accepted no-name-change', st.filter(pl.col('acc9') & (pl.col('bnc') == ''))['sfx'].mean(),
      '| accepted all', st.filter(pl.col('acc9'))['sfx'].mean(),
      '| rejected SWAP desc (decoys)', st.filter(~pl.col('acc9') & (pl.col('key') == 'SWAP|n_swap:desc|NSAME'))['sfx'].mean(),
      '| rejected SWAP other', st.filter(~pl.col('acc9') & (pl.col('key') == 'SWAP|n_swap:other|NSAME'))['sfx'].mean(),
      '| rejected LEGAL_CHANGE+SWAP', st.filter(~pl.col('acc9') & (pl.col('bnc') == 'LEGAL_CHANGE+SWAP'))['sfx'].mean(),
      '| rejected ADD+SWAP', st.filter(~pl.col('acc9') & (pl.col('bnc') == 'ADD+SWAP'))['sfx'].mean())
print(st.filter(~pl.col('acc9')).group_by('key').agg(n=pl.len(), sfx=pl.col('sfx').mean(), low=pl.col('lower').mean(), st2=(pl.col('sm2') == 'st').mean(),
      g=pl.col('p2g').median(), t=pl.col('p3').median()).filter(pl.col('n') >= 60).sort('n', descending=True).head(40))
print('NMISS street-word match (rejected):')
print(d.filter(m & ~pl.col('acc9')).group_by('key').agg(n=pl.len(), stw1=(pl.col('stw') >= 0.99).mean(), stw0=(pl.col('stw') == 0).mean(), low=pl.col('lower').mean(),
      low_stw1=pl.col('lower').filter(pl.col('stw') >= 0.99).mean(), n_stw1=(pl.col('stw') >= 0.99).sum()).filter(pl.col('n') >= 60).sort('n', descending=True).head(30))
