"""France check: run the same generator on ALL France test records (matched and unmatched in v10b). For records v10b matched,
how often does the generator's unique S1 equal v10b's S1? Also: how common are acronym-name records among v10b France matches
vs among US/India truth (is France under-matching acronyms?)."""
import os, sys, time
sys.path.insert(0, r"C:/ber_scratch/final/same-address")
from salib import *
from common import WORK, id_to_int, log, read_tsv
OUT = r"C:/ber_scratch/final/same-address"
R = os.path.dirname(WORK)
t0 = time.time()


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).filter(
    pl.col("country") == "France").select(s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
rec = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select("entity_id", "business_name", "business_address", "country") for k in (2, 3)])
         .filter(pl.col("country") == "France")
         .select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address", country="country").collect())
rec = add_key(rec, "qa")
s1k = add_key(s1, "sa")
j = generate(rec.select("q", "qn", "qa", "country", "k"), s1k.select("s", "sn", "sa", "country", "k"))
vq = v.select("q", v_s="s").group_by("q").agg(pl.col("v_s").first())
k = j.filter(pl.col("keep")).join(vq, on="q", how="left").with_columns(
    nm=pl.col("nops").str.extract_all(r"n_[a-z_]+").list.join("+"), acr=pl.col("nops").str.contains("n_acronym"))
m = k.filter(pl.col("v_s").is_not_null())
log(f"France records {rec.height}; generator keep {k.height}; of those matched in v10b {m.height}; agreement with v10b S1 {(m['v_s'] == m['s']).mean():.4f}")
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
log(str(k.with_columns(matched=pl.col("v_s").is_not_null(), agree=pl.col("v_s") == pl.col("s")).group_by("acr").agg(
    n=pl.len(), matched=pl.col("matched").mean(), agree_if_matched=pl.col("agree").drop_nulls().mean())))
log(str(k.filter(pl.col("v_s").is_not_null()).with_columns(agree=pl.col("v_s") == pl.col("s")).group_by("nm").agg(
    n=pl.len(), agree=pl.col("agree").mean()).sort("n", descending=True).head(20)))
dis = m.filter(pl.col("v_s") != pl.col("s")).join(s1.select(v_s="s", vsn="sn", vsa="sa"), on="v_s")
log("disagreements (generator S1 vs v10b S1):\n" + str(dis.select("qn", "qa", "sn", "sa", "vsn", "vsa").head(20)))
# acronym-looking record names: share among v10b-matched vs unmatched France records
rec = rec.with_columns(acrlike=pl.col("qn").str.contains(r"^[A-Z]{2,6}$") | pl.col("qn").str.contains(r"^[A-Z]{2,6} (SARL|SAS|SASU|SA|EURL|SCI|SNC)$"))
rec = rec.join(vq, on="q", how="left")
log(f"France records acronym-like: {rec['acrlike'].sum()}; matched in v10b {rec.filter(pl.col('acrlike'))['v_s'].is_not_null().mean():.3f} "
    f"vs all France records matched {rec['v_s'].is_not_null().mean():.3f}")
k.write_parquet(os.path.join(OUT, "fr_all_keep.parquet"))
log(f"done {time.time()-t0:.0f}s")
