import sys, os
sys.path.insert(0, 'E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src')
import polars as pl
from common import WORK, id_to_int
from pipeline import decide_expf
OUT = 'C:/ber_scratch/frfix2/fn/'
A = pl.read_parquet(OUT + 'fn_add_all.parquet').filter(pl.col('tier').str.starts_with('A'))
add = A.select('q', 's').with_columns(add=pl.lit(True))
add.write_parquet('E:/Projects/Amazon ML Challenge/work/frfix2/fn_add_set.parquet')
sc = pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet')
n0 = sc.height
sc = sc.join(add, on=['q', 's'], how='left').with_columns(
    p2=pl.when(pl.col('add').fill_null(False)).then(pl.lit(0.95, dtype=pl.Float32)).otherwise(pl.col('p2')).cast(sc['p2'].dtype)).drop('add')
assert sc.height == n0
print('pairs set to 0.95:', sc.join(add, on=['q', 's']).height, 'of', add.height)
dst = 'E:/Projects/Amazon ML Challenge/work/frfix2/test_scores_v9b_fnadd.parquet'
sc.write_parquet(dst)
print('wrote', dst, sc.height)
# simulate the France decision exactly like finalize.py (legal-form veto + expF floor 0.5 alpha 1)
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'))
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]
def forms(files):
    d = pl.concat([pl.read_parquet(f, columns=["entity_id", "business_name", "country"]) for f in files]).filter(pl.col("country") == "France")
    t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
           .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique())
fs = forms([f'{WORK}/test_s1.parquet']).rename({'id': 's', 'f': 'fs'})
fq = forms([f'{WORK}/test_s{k}.parquet' for k in (2, 3)]).rename({'id': 'q', 'f': 'fq'})
def decide(scores):
    B = scores.join(s1, on='s').join(fs, on='s', how='left').join(fq, on='q', how='left')
    veto = ((pl.col('fs').list.len() > 0) & (pl.col('fq').list.len() > 0) & (pl.col('fs').list.set_intersection('fq').list.len() == 0)).fill_null(False)
    B = B.with_columns(p2=pl.when(veto).then(0.0).otherwise(pl.col('p2')).cast(pl.Float32)).drop('fs', 'fq')
    return decide_expf(B, 'p2', 0.5, 1.0)
old = decide(pl.read_parquet(f'{WORK}/frfix/test_scores_frmin_descveto_gnadd.parquet', columns=['q', 's', 'p2']))
new = decide(sc.select('q', 's', 'p2'))
v9 = pl.read_parquet(OUT + 'v9b_fr_acc.parquet')
print('v9b France pairs: file', v9.height, '| re-simulated', old.height, '| same', old.join(v9, on=['s', 'q']).height)
gained = new.join(old, on=['s', 'q'], how='anti'); lost = old.join(new, on=['s', 'q'], how='anti')
print('new France pairs', new.height, '| added', gained.height, '| removed', lost.height,
      '| added that are in the add set', gained.join(add, on=['q', 's']).height)
gained.write_parquet(OUT + 'sim_added.parquet'); lost.write_parquet(OUT + 'sim_removed.parquet')
# macro F0.5 change estimate: per added pair, current predicted count c of its S1 (v9b), gain if true / loss if false
c = old.group_by('s').agg(c=pl.len())
g = gained.join(A.select('q', 's', 'tier'), on=['q', 's'], how='left').join(c, on='s', how='left').with_columns(pl.col('c').fill_null(0))
print(g.group_by('tier', pl.col('c').clip(0, 3)).agg(n=pl.len()).sort('tier', 'c'))
g.write_parquet(OUT + 'sim_added_c.parquet')
