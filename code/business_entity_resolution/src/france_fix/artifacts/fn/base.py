# Base table: France best candidates (fr_top) + census ops + v9b decision status
import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
t = time.time()
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).filter(pl.col('country') == 'France')
fr_ids = set(s1['entity_id'].to_list())
rows = []
with open(f'{ROOT}/submissions/v9b/matching_results.tsv', encoding='utf8') as f:
    f.readline()
    for ln in f:
        a, b = ln.rstrip('\n').split('\t')
        if b and a in fr_ids:
            for x in b.split(','):
                rows.append((a, x))
acc = pl.DataFrame(rows, schema=['s1', 'q1'], orient='row').select(s=id_to_int('s1'), q=id_to_int('q1'))
print('v9b France accepted pairs', acc.height, 'records', acc['q'].n_unique())
acc.write_parquet(f'{SCRATCH}/frfix2/fn/v9b_fr_acc.parquet')
d = pl.read_parquet(f'{SCRATCH}/france/fr_top.parquet',
                    columns=['q', 's', 'p2', 'p1', 'p2g', 'qn', 'qa', 'sn', 'sa', 'pat', 's_2', 'p2_2', 'sn_2', 'sa_2', 'acc'])
o = pl.read_parquet(f'{SCRATCH}/frfix2/census/fr_ops.parquet', columns=['q', 's', 'ops', 'nc', 'num'])
d = d.join(o, on=['q', 's'], how='left')
# v9b score of the pair
sc = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet', columns=['q', 's', 'p2']).rename({'p2': 'p9'})
sc = sc.filter(pl.col('q').is_in(d['q'].implode()))
d = d.join(sc, on=['q', 's'], how='left')
# v9b best score of the record over all candidates
bestq = sc.sort('p9', descending=True).unique('q', keep='first').rename({'s': 's9best', 'p9': 'p9best'})
d = d.join(bestq, on='q', how='left')
# stage-3 transformer prob
p3 = pl.read_parquet(f'{WORK}/test_scores_blend_ab_a2.parquet', columns=['q', 's', 'p2']).rename({'p2': 'p3'})
p3 = p3.filter(pl.col('q').is_in(d['q'].implode()))
d = d.join(p3, on=['q', 's'], how='left')
d = d.join(acc.with_columns(acc9=pl.lit(True)), on=['q', 's'], how='left').with_columns(pl.col('acc9').fill_null(False))
qa9 = acc.group_by('q').agg(s9acc=pl.col('s').first())
d = d.join(qa9, on='q', how='left').with_columns(q_acc_else=pl.col('s9acc').is_not_null() & ~pl.col('acc9'))
d = d.with_columns(sig=pl.col('nc').list.join('+') + '|' + pl.col('num'))
d.write_parquet(f'{SCRATCH}/frfix2/fn/fr_base.parquet')
print(d.height, 'acc9', d['acc9'].sum(), 'q_acc_else', d['q_acc_else'].sum(), 'v7ens acc', d['acc'].sum(), 'sec', round(time.time() - t))
print('accepted v9b pairs not in fr_top:', acc.join(d.select('q', 's'), on=['q', 's'], how='anti').height)
