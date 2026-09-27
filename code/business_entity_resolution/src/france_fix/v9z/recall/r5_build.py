# Final France recall add set + scores file (current v9y France scores; added pairs at p2 = 0.95), with decision simulation.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
from pipeline import decide_expf
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(45)
R = f'{SCRATCH}/frfix3/recall/'
FN = f'{SCRATCH}/frfix2/fn/'
OUTW = f'{WORK}/frfix3/'
acc = pl.read_parquet(R + 'acc_v9y_fr.parquet')
# (b) earlier missed-true set, tiers A1-A4
b = pl.read_parquet(f'{WORK}/frfix2/fn_add_set_final.parquet').join(pl.read_parquet(FN + 'fn_add_all.parquet', columns=['q', 's', 'tier']), on=['q', 's'])
b = b.with_columns(grp=pl.lit('b_') + pl.col('tier'), est=pl.col('tier').replace_strict({'A1': 0.91, 'A2': 0.96, 'A3': 0.92, 'A4': 0.92}, return_dtype=pl.Float64))
# (c) only-true name changes at the same address (acronym, web domain, dotted legal form): name fits, no sibling S1 at the address fits
c = pl.read_parquet(R + 'c_cands.parquet').filter(pl.col('fit') & (pl.col('namb') == 0))
EST_C = {'ACRONYM||NSAME': 0.95, 'ACRONYM+LEGAL_DROP||NSAME': 0.95, 'DOMAIN||NSAME': 0.93, 'LEGAL_DOT||NSAME': 0.95,
         'ACRONYM||NMISS': 0.85, 'ACRONYM+LEGAL_DROP||NMISS': 0.85, 'DOMAIN||NMISS': 0.85, 'LEGAL_DOT||NMISS': 0.9}
c = c.select('q', 's', grp=pl.lit('c_') + pl.col('key'), est=pl.col('key').replace_strict(EST_C, return_dtype=pl.Float64))
# (a) vetoed records with an exact-name S1 at the same address (or same street, record number missing)
a = pl.read_parquet(R + 'veto_alt_wd.parquet').filter((pl.col('addr') == 'same') & (pl.col('wdiff') == 0))
print('(a) exact-name twins at the same address:'); print(a.select('qn', 'xn', 'qa', 'xa', 'addr', 'px'))
a = a.select('q', s='sx', grp=pl.lit('a_twin'), est=pl.lit(0.9))
add = pl.concat([b.select('q', 's', 'grp', 'est'), c, a])
add = add.filter(~pl.col('q').is_in(acc['q'].implode())).unique('q', keep='first', maintain_order=True)
# France legal-form veto that finalize applies (BER_FR_LEGAL_VETO=1): record and S1 both carry forms and share none -> p2 0
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]
def forms(files):
    d = pl.concat([pl.read_parquet(f, columns=["entity_id", "business_name", "country"]) for f in files]).filter(pl.col("country") == "France")
    t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
           .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique())
fs = forms([f'{WORK}/test_s1.parquet']).rename({'id': 's', 'f': 'fs'})
fq = forms([f'{WORK}/test_s{k}.parquet' for k in (2, 3)]).rename({'id': 'q', 'f': 'fq'})
s1 = pl.read_parquet(f'{WORK}/test_s1.parquet', columns=['entity_id', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'))
lv = add.join(fs, on='s', how='left').join(fq, on='q', how='left').with_columns(
    lveto=((pl.col('fs').list.len() > 0) & (pl.col('fq').list.len() > 0) & (pl.col('fs').list.set_intersection('fq').list.len() == 0)).fill_null(False))
print('legal-form veto would kill', lv['lveto'].sum(), 'adds:'); print(lv.group_by('grp').agg(n=pl.len(), lveto=pl.col('lveto').sum()).sort('grp'))
add = lv.filter(~pl.col('lveto')).select('q', 's', 'grp', 'est')
assert add.join(s1, on='s').height == add.height
add.write_parquet(OUTW + 'recall_add_set.parquet')
# scores file
sc = pl.read_parquet(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet')
n0 = sc.height
sc = sc.join(add.select('q', 's', a=pl.lit(True)), on=['q', 's'], how='left').with_columns(
    p2=pl.when(pl.col('a').fill_null(False)).then(pl.lit(0.95)).otherwise(pl.col('p2')).cast(pl.Float32)).drop('a')
new = add.join(sc.select('q', 's'), on=['q', 's'], how='anti').select('q', 's', p1=pl.lit(0.95, pl.Float32), p2=pl.lit(0.95, pl.Float32))
print('adds in existing score rows', add.height - new.height, '| new rows appended', new.height)
sc = pl.concat([sc, new.select(sc.columns).cast(sc.schema)])
assert sc.height == n0 + new.height and sc.select('q', 's').is_duplicated().sum() == 0
dst = OUTW + 'test_scores_v9y_recall.parquet'
sc.write_parquet(dst)
print('wrote', dst, sc.height)
def decide(scores):
    B = scores.join(s1, on='s').join(fs, on='s', how='left').join(fq, on='q', how='left')
    veto = ((pl.col('fs').list.len() > 0) & (pl.col('fq').list.len() > 0) & (pl.col('fs').list.set_intersection('fq').list.len() == 0)).fill_null(False)
    B = B.with_columns(p2=pl.when(veto).then(0.0).otherwise(pl.col('p2')).cast(pl.Float32)).drop('fs', 'fq')
    return decide_expf(B, 'p2', 0.5, 1.0)
old = decide(pl.read_parquet(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet', columns=['q', 's', 'p2']).join(s1, on='s'))
nw = decide(sc.select('q', 's', 'p2').join(s1, on='s'))
print('v9y France pairs: file', acc.height, '| re-simulated old', old.height, '| same', old.join(acc, on=['s', 'q']).height)
gained = nw.join(old, on=['s', 'q'], how='anti'); lost = old.join(nw, on=['s', 'q'], how='anti')
lost.join(sc.select('q','s','p2'),on=['q','s']).join(add.select('s').unique().with_columns(sadd=pl.lit(True)),on='s',how='left').write_parquet(R+'sim_removed.parquet'); print('removed pairs: S1 got an add', lost.join(add.select('s').unique(),on='s').height, 'record got an add', lost.join(add.select('q'),on='q').height); print('new', nw.height, '| added', gained.height, '| removed', lost.height, '| added in add set', gained.join(add, on=['q', 's']).height)
# expected macro-F0.5 change per S1 row (France S1 = 259,452): current predicted count k of the S1
k = old.group_by('s').agg(k=pl.len())
g = gained.join(add, on=['q', 's'], how='left').join(k, on='s', how='left').with_columns(pl.col('k').fill_null(0))
g = g.with_columns(kk=pl.col('k').cast(pl.Float64))
g = g.with_columns(gain=pl.when(pl.col('kk') == 0).then(1.0).otherwise(1 - 1.25 * pl.col('kk') / (1.25 * pl.col('kk') + 0.25)),
                   loss=pl.when(pl.col('kk') == 0).then(1.0).otherwise(1 - 1.25 * pl.col('kk') / (1.25 * pl.col('kk') + 1)))
g = g.with_columns(dF=pl.col('est') * pl.col('gain') - (1 - pl.col('est')) * pl.col('loss'), be=pl.col('loss') / (pl.col('gain') + pl.col('loss')))
NS = 259452
print(g.group_by('grp').agg(n=pl.len(), est=pl.col('est').first(), k0=(pl.col('k') == 0).sum(), be=pl.col('be').mean(), dF=(pl.col('dF').sum() / NS),
      dF_at_0p80=((0.80 * pl.col('gain') - 0.20 * pl.col('loss')).sum() / NS)).sort('grp'))
print('total France dF (est)', round(g['dF'].sum() / NS, 5), '| LB', round(0.15 * g['dF'].sum() / NS, 6))
for pp in (0.7, 0.76, 0.8, 0.85, 0.9, 0.95):
    print(f'  if all adds were {pp:.2f} true: France {((pp * g["gain"] - (1 - pp) * g["loss"]).sum() / NS):+.5f}')
