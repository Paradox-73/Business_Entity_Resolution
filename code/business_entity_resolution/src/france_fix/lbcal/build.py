import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
import france_cal as fc
OUT = f"{SCRATCH}/frfix/lbcal"
FR = f"{SCRATCH}/france"
t0 = time.time()
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(s=id_to_int("entity_id"))
def load(f, name):
    return pl.scan_parquet(f).select("q", "s", pl.col("p2").alias(name)).join(s1.lazy(), on="s").collect()
b = load(os.path.join(WORK, "test_scores_full_cons.parquet"), "g")
b = b.join(load(os.path.join(WORK, "test_scores_blend_ab_a2.parquet"), "p3"), on=["q", "s"], how="left")
b = b.join(load(os.path.join(WORK, "ce", "test_scores_blend_ab_a2_frmin.parquet"), "pens"), on=["q", "s"], how="left")
b = b.join(load(os.path.join(WORK, "ce", "test_scores_blend_ab_a2_frrest.parquet"), "pm"), on=["q", "s"], how="left")
print("France pairs", b.height, "records", b["q"].n_unique(), time.time() - t0, flush=True)
# legal-form veto flag (same as finalize.py)
FORMS = fc.FR_FORMS
def forms(files):
    d = pl.concat([pl.read_parquet(f, columns=["entity_id", "business_name", "country"]).filter(pl.col("country") == "France") for f in files])
    t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
           .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FORMS))).list.unique())
fs = forms([os.path.join(WORK, "test_s1.parquet")]).rename({"id": "s", "f": "fs"})
fq = forms([os.path.join(WORK, f"test_s{k}.parquet") for k in (2, 3)]).rename({"id": "q", "f": "fq"})
b = b.join(fs, on="s", how="left").join(fq, on="q", how="left")
b = b.with_columns(veto=((pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0) & (pl.col("fs").list.set_intersection("fq").list.len() == 0)).fill_null(False)).drop("fs", "fq")
print("veto pairs", b["veto"].sum(), time.time() - t0, flush=True)
b = b.with_columns(pe=pl.when(pl.col("veto")).then(0.0).otherwise(pl.col("pens")))
b = b.sort(["q", "pe", "g"], descending=[False, True, True]).with_columns(rk=pl.int_range(1, pl.len() + 1).over("q"))
# version pair sets
def tsv_pairs(p):
    sub = pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0)
    return (sub.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids")).join(s1, on="s"))
V = {"ens": pl.read_parquet(f"{FR}/pairs_v7ens.parquet"), "i": pl.read_parquet(f"{FR}/pairs_v7i.parquet"), "j": pl.read_parquet(f"{FR}/pairs_v7j.parquet"),
     "gn": pl.read_parquet(f"{FR}/pairs_v7g_num.parquet"), "m": tsv_pairs(rf"{ROOT}/submissions/v7m/matching_results.tsv"),
     "v3": tsv_pairs(rf"{ROOT}/submissions/v3/matching_results.tsv")}
for k, v in V.items():
    b = b.join(v.select(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).with_columns(pl.lit(True).alias("in_" + k)), on=["q", "s"], how="left").with_columns(pl.col("in_" + k).fill_null(False))
    print(k, v.height, "found in cand", b["in_" + k].sum(), flush=True)
anyv = pl.any_horizontal([pl.col("in_" + k) for k in V])
keep = b.filter((pl.col("rk") <= 3) | anyv)
print("kept", keep.height, "of", b.height, "rank<=3", (b["rk"] <= 3).sum(), time.time() - t0, flush=True)
# also save rank>3 aggregate mass per record/S1 (g) for recall term
rest = b.filter(~((pl.col("rk") <= 3) | anyv)).select("q", "s", "g", "p3", "veto")
rest.write_parquet(f"{OUT}/rest_pairs.parquet")
del b
# patterns: reuse fr_top
top = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "pat", "s_2", "pat_2"])
pp = pl.concat([top.select("q", "s", "pat"), top.filter(pl.col("s_2").is_not_null()).select("q", s="s_2", pat="pat_2")]).unique(["q", "s"])
keep = keep.join(pp, on=["q", "s"], how="left")
need = keep.filter(pl.col("pat").is_null()).select("q", "s")
print("need pattern", need.height, flush=True)
d = fc.attach_text(need, "test").select("q", "s", pat2="pat")
keep = keep.join(d, on=["q", "s"], how="left").with_columns(pat=pl.coalesce("pat", "pat2")).drop("pat2")
# street same (fr_restore)
import fr_restore as frr
from rapidfuzz import fuzz
sa = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "business_address", "country"]).filter(pl.col("country") == "France").select(s=id_to_int("entity_id"), sa="business_address")
qa = pl.concat([pl.scan_parquet(f"{WORK}/test_s{k}.parquet").select(q=id_to_int("entity_id"), qa="business_address").join(keep.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
keep = keep.join(sa, on="s", how="left").join(qa, on="q", how="left")
ss = [frr.street(a) for a in keep["sa"].to_list()]
qs = [frr.street(a) for a in keep["qa"].to_list()]
ok = [bool(a[0] is not None and a[0] == c[0] and a[1] and c[1] and fuzz.token_sort_ratio(a[1], c[1]) >= 85) for a, c in zip(ss, qs)]
keep = keep.with_columns(samestreet=pl.Series(ok)).drop("sa", "qa")
keep.write_parquet(f"{OUT}/pairs.parquet")
print("wrote", keep.height, time.time() - t0)
