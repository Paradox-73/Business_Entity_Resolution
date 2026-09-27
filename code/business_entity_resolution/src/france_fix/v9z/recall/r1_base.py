# Base: France current scores (v9y France), current decision, veto set status, candidate lists of vetoed records.
import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
t = time.time()
R = f'{SCRATCH}/frfix3/recall/'
TW = f'{SCRATCH}/frfix/twins/'
s1 = pl.read_parquet(TW + 'fr_s1k.parquet')
fr_s = set(s1['s'].to_list())
# current decision = v9y France rows
rows = []
from common import int_to_id
s1ids = set(pl.Series(s1['s']).to_list())
with open(f'{ROOT}/submissions/v9y/matching_results.tsv', encoding='utf8') as f:
    f.readline()
    for ln in f:
        a, b = ln.rstrip('\n').split('\t')
        if b:
            for x in b.split(','):
                rows.append((a, x))
acc = pl.DataFrame(rows, schema=['s1', 'q1'], orient='row').select(s=id_to_int('s1'), q=id_to_int('q1'))
acc = acc.filter(pl.col('s').is_in(s1['s'].implode()))
print('v9y France accepted pairs', acc.height, 'records', acc['q'].n_unique(), 'S1', acc['s'].n_unique())
acc.write_parquet(R + 'acc_v9y_fr.parquet')
Q = pl.read_parquet(TW + 'fr_recsk.parquet', columns=['q'])
sc = (pl.scan_parquet(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet').select('q', 's', 'p2')
      .join(Q.lazy(), on='q').collect())
p3 = (pl.scan_parquet(f'{WORK}/test_scores_blend_ab_a2.parquet').select('q', 's', pl.col('p2').alias('p3'))
      .join(Q.lazy(), on='q').collect())
g = (pl.scan_parquet(f'{WORK}/test_scores_full_cons.parquet').select('q', 's', pl.col('p2').alias('g'))
     .join(Q.lazy(), on='q').collect())
sc = sc.join(p3, on=['q', 's'], how='left').join(g, on=['q', 's'], how='left')
sc = sc.join(acc.with_columns(acc=pl.lit(True)), on=['q', 's'], how='left').with_columns(pl.col('acc').fill_null(False))
print('France scored pairs', sc.height, 'records', sc['q'].n_unique(), 'accepted in scores', sc['acc'].sum(), 'sec', round(time.time() - t))
sc.write_parquet(R + 'fr_scores.parquet')
v = pl.read_parquet(f'{SCRATCH}/frfix/namechg/veto_set.parquet')
qacc = acc.group_by('q').agg(s_now=pl.col('s').first(), n_now=pl.len())
v = v.join(qacc, on='q', how='left')
print('veto set', v.height, 'records', v['q'].n_unique(), 'currently matched', v['s_now'].is_not_null().sum(),
      'matched to vetoed s', (v['s_now'] == v['s']).sum())
print(v['kind'].value_counts(), v['num'].value_counts())
v.write_parquet(R + 'veto_status.parquet')
