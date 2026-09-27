"""v9b France decision (legal veto + decide_expf floor .5 alpha 1 on p2), same as finalize.py; saves accepted pairs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
from pipeline import decide_expf

OUT = f'{SCRATCH}/frfix2/fp/'
V9B = os.path.join(WORK, 'frfix', 'test_scores_frmin_descveto_gnadd.parquet')
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]

s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).filter(
    pl.col('country') == 'France').select(s=id_to_int("entity_id"))


def forms(split_files):
    d = pl.concat([pl.read_parquet(f, columns=["entity_id", "business_name", "country"]) for f in split_files])
    d = d.filter(pl.col("country") == "France")
    t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
           .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(
        pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique())


def france_scores(path):
    b = pl.read_parquet(path, columns=['q', 's', 'p2']).join(s1, on='s')
    fs = forms([os.path.join(WORK, "test_s1.parquet")]).rename({"id": "s", "f": "fs"})
    fq = forms([os.path.join(WORK, f"test_s{k}.parquet") for k in (2, 3)]).rename({"id": "q", "f": "fq"})
    b = b.join(fs, on="s", how="left").join(fq, on="q", how="left")
    veto = ((pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0)
            & (pl.col("fs").list.set_intersection("fq").list.len() == 0)).fill_null(False)
    return b.with_columns(p2=pl.when(veto).then(0.0).otherwise(pl.col("p2")).cast(pl.Float32)).drop("fs", "fq")


if __name__ == '__main__':
    b = france_scores(V9B)
    b.write_parquet(OUT + 'fr_scores_v9b_lv.parquet')
    m = decide_expf(b, 'p2', 0.5, 1.0)
    m = m.join(b, on=['s', 'q'], how='left')
    m.write_parquet(OUT + 'acc_v9b.parquet')
    print('France scored pairs', b.height, 'records', b['q'].n_unique(), 'accepted', m.height)
    # cross-check against the production output
    t = pl.read_csv(os.path.join(WORK, 'frfix', 'out_descveto_gnadd_all', 'matching_results.tsv'), separator='\t')
    t = (t.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids')
          .filter(pl.col('matched_entity_ids') != '')
          .select(s=id_to_int('source1_entity_id'), q=id_to_int('matched_entity_ids')).join(s1, on='s'))
    print('production TSV France pairs', t.height, 'common', t.join(m, on=['s', 'q']).height)
