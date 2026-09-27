# (a) For each vetoed record: other S1s (candidate list, same address key anywhere in France, same token-set name in city)
import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import name_ops, words
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(330); pl.Config.set_fmt_str_lengths(45)
t = time.time()
R = f'{SCRATCH}/frfix3/recall/'
TW = f'{SCRATCH}/frfix/twins/'
v = pl.read_parquet(R + 'veto_status.parquet', columns=['q', 's', 'num', 'kind', 'p2', 'p3', 'p2g'])
V = v.select('q')
rec = pl.read_parquet(TW + 'fr_recsk.parquet', columns=['q', 'qn', 'qa', 'qnum', 'qst', 'qcity']).join(V, on='q')
s1 = pl.read_parquet(TW + 'fr_s1k.parquet')
kk = pl.read_parquet(TW + 'fr_s1kk.parquet', columns=['s', 'ks'])
s1 = s1.join(kk, on='s')
sc = pl.read_parquet(R + 'fr_scores.parquet')
acc = pl.read_parquet(R + 'acc_v9y_fr.parquet')
nacc = acc.group_by('s').agg(k_now=pl.len())
# 1. candidate-list alternatives
c1 = sc.join(V, on='q').join(v.select('q', sv='s'), on='q').filter(pl.col('s') != pl.col('sv')).select('q', sx='s').with_columns(src_c=pl.lit(True))
# 2. same address key (city + street + number) anywhere in France
ak = rec.filter(pl.col('qnum').is_not_null() & (pl.col('qnum') != '') & pl.col('qst').is_not_null() & (pl.col('qst') != ''))
c2 = ak.select('q', 'qcity', 'qst', 'qnum').join(s1.select(sx='s', qcity='scity', qst='sst', qnum='snum'), on=['qcity', 'qst', 'qnum']).select('q', 'sx').with_columns(src_a=pl.lit(True))
# 3. same token-set name in same city
tw = pl.scan_parquet(TW + 'twins_all.parquet').select('q', 'sx').join(V.lazy(), on='q').collect().with_columns(src_n=pl.lit(True))
c = pl.concat([c1.select('q', 'sx'), c2.select('q', 'sx'), tw.select('q', 'sx')]).unique()
c = c.join(v.select('q', sv='s'), on='q').filter(pl.col('sx') != pl.col('sv'))
c = c.join(c1, on=['q', 'sx'], how='left').join(c2, on=['q', 'sx'], how='left').join(tw, on=['q', 'sx'], how='left').with_columns(
    pl.col('src_c', 'src_a', 'src_n').fill_null(False))
print('alternative S1 pairs', c.height, 'records with any', c['q'].n_unique(), 'of', v.height)
print(c.group_by('src_c', 'src_a', 'src_n').agg(n=pl.len(), nq=pl.col('q').n_unique()).sort('n', descending=True))
c = c.join(rec, on='q').join(s1.select(sx='s', xn='sn', xa='sa', xnum='snum', xst='sst', xcity='scity', xks='ks'), on='sx')
c = c.join(sc.select('q', sx='s', px='p2', p3x='p3', gx='g'), on=['q', 'sx'], how='left').join(nacc.rename({'s': 'sx', 'k_now': 'kx_now'}), on='sx', how='left').with_columns(pl.col('kx_now').fill_null(0))
# name ops record vs alternative S1
qn, xn = c['qn'].to_list(), c['xn'].to_list()
nops = [sorted(name_ops(a, b)) for a, b in zip(qn, xn)]
c = c.with_columns(nops=pl.Series(nops, dtype=pl.List(pl.Utf8)))
c = c.with_columns(same_num=(pl.col('qnum') == pl.col('xnum')), same_st=(pl.col('qst') == pl.col('xst')), same_city=(pl.col('qcity') == pl.col('xcity')),
                   nn=pl.col('nops').list.len(), lower=pl.col('nops').list.contains('n_lower'))
c.write_parquet(R + 'veto_alt.parquet')
print('sec', round(time.time() - t))
print(c.group_by('same_city', 'same_st', 'same_num').agg(n=pl.len(), nq=pl.col('q').n_unique(), n0=(pl.col('nn') == 0).sum(), n1=(pl.col('nn') == 1).sum()).sort('n', descending=True))
