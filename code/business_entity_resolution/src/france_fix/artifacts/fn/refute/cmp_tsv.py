import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # src/
from common import ROOT, WORK  # noqa: E402
import polars as pl
from common import WORK, id_to_int
fr = set(pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).filter(pl.col('country') == 'France')['entity_id'].to_list())
def pairs(path, only_fr=True):
    rows = []
    with open(path, encoding='utf8') as f:
        f.readline()
        for ln in f:
            a, b = ln.rstrip('\n').split('\t')
            if b and ((a in fr) == only_fr):
                rows += [(a, x) for x in b.split(',')]
    return pl.DataFrame(rows, schema=['s', 'q'], orient='row')
new = pairs(f'{WORK}/frfix2/refute_fn_out/matching_results.tsv')
old = pairs(f'{ROOT}/submissions/v9b/matching_results.tsv')
add = new.join(old, on=['s', 'q'], how='anti'); rem = old.join(new, on=['s', 'q'], how='anti')
print('France pairs v9b', old.height, 'new', new.height, 'added', add.height, 'removed', rem.height)
aset = pl.read_parquet(f'{WORK}/frfix2/fn_add_set.parquet')
addi = add.select(s=id_to_int('s'), q=id_to_int('q'))
print('added pairs in add set:', addi.join(aset, on=['q', 's']).height)
# US/India of v9b vs this build (should be v7ens US/India, since v9b's scores = v7ens US/India)
nu = pairs(f'{WORK}/frfix2/refute_fn_out/matching_results.tsv', only_fr=False)
ou = pairs(f'{ROOT}/submissions/v9b/matching_results.tsv', only_fr=False)
print('US/India pairs v9b', ou.height, 'this build', nu.height, 'diff', nu.join(ou, on=['s', 'q'], how='anti').height, ou.join(nu, on=['s', 'q'], how='anti').height)
rem.write_parquet('removed37.parquet')
