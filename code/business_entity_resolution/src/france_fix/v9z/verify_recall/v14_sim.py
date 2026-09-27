# Independent France decision simulation (finalize's France path: legal-form veto, expF floor 0.5 alpha 1) on old and new scores
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
from pipeline import decide_expf
V = f'{SCRATCH}/frfix3/verify_recall/'
s1 = pl.read_parquet(os.path.join(WORK, 'test_s1.parquet'), columns=['entity_id', 'country']).filter(pl.col('country') == 'France').select(s=id_to_int('entity_id'))
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]
def forms(split_files):
    d = pl.concat([pl.read_parquet(f, columns=["entity_id", "business_name", "country"]) for f in split_files]).filter(pl.col("country") == "France")
    t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
           .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique())
fs = forms([os.path.join(WORK, "test_s1.parquet")]).rename({"id": "s", "f": "fs"})
fq = forms([os.path.join(WORK, f"test_s{k}.parquet") for k in (2, 3)]).rename({"id": "q", "f": "fq"})
def run(path):
    B = pl.read_parquet(path, columns=['q', 's', 'p2']).join(s1, on='s').join(fs, on='s', how='left').join(fq, on='q', how='left')
    veto = ((pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0) & (pl.col("fs").list.set_intersection("fq").list.len() == 0)).fill_null(False)
    B = B.with_columns(p2=pl.when(veto).then(0.0).otherwise(pl.col("p2")).cast(pl.Float32)).drop("fs", "fq")
    return decide_expf(B, 'p2', 0.5, 1.0)
old = run(f'{WORK}/frfix2/test_scores_v9b_fpveto.parquet')
new = run(f'{WORK}/frfix3/test_scores_v9y_recall.parquet')
v9 = pl.read_parquet(V + 'fr_v9y.parquet')
print('old sim', old.height, '| v9y file', v9.height, '| equal pairs', old.join(v9, on=['s', 'q']).height)
gained = new.join(old, on=['s', 'q'], how='anti'); lost = old.join(new, on=['s', 'q'], how='anti')
add = pl.read_parquet(f'{WORK}/frfix3/recall_add_set.parquet')
print('new', new.height, 'gained', gained.height, 'lost', lost.height, 'gained in add set', gained.join(add, on=['s', 'q']).height, 'add pairs not gained', add.join(new, on=['s','q'], how='anti').height)
new.write_parquet(V + 'fr_sim_new.parquet')
# which veto removed the 11 add pairs that v7ens had?
v7 = pl.read_parquet(V + 'fr_v7ens.parquet')
x = add.join(v7, on=['s', 'q'])
veto = pl.read_parquet(f'{SCRATCH}/frfix/namechg/veto_set.parquet', columns=['q', 's']).with_columns(dv=pl.lit(True))
fp = pl.read_parquet(f'{WORK}/frfix2/fp_veto_set.parquet').select('q', 's').with_columns(fpv=pl.lit(True))
print(x.join(veto, on=['q', 's'], how='left').join(fp, on=['q', 's'], how='left'))
