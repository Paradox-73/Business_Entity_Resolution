"""fr-committee step 1: France pairs Gathik matched, inside our France lists, not in v10b.
Split add (record unmatched in v10b) / move (record matched in v10b to another S1). Remove vetoes + legal conflicts.
Attach ops, lowercase flag, our p2 (combo), gathik p2 (frmin), gathik xgb p2."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from common import WORK, id_to_int, log, read_tsv
from pipeline import decide_expf
from ops import ops

R = os.path.dirname(WORK)
T = rf"{SCRATCH}/final/fr-committee"
TMP = rf"{SCRATCH}"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
fr = s1.filter(pl.col("country") == "France").drop("country")
frs = fr.select("s")
log(f"France S1 {fr.height}")

v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
vfr = v.join(frs, on="s")
log(f"v10b pairs {v.height}; France {vfr.height}")
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv")).join(frs, on="s")
log(f"gathik France pairs {g.height}")

combo = i64(pl.scan_parquet(os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet")).select("q", "s", "p2").join(frs.lazy(), on="s").collect())
log(f"combo France rows {combo.height}")
# verify v10b France = decide(combo) + recall_added
added = pl.read_parquet(os.path.join(WORK, "out_v10b_fr", "france_recall_added.parquet"), columns=["s", "q"])
for fl in (0.5,):
    dd = decide_expf(combo, "p2", fl, 1.0)
    base = vfr.join(added, on=["s", "q"], how="anti")
    log(f"re-decide floor {fl}: {dd.height} pairs; v10b France minus recall {base.height}; common {dd.join(base, on=['s','q']).height}")

lists = pl.concat([
    combo.select("q", "s"),
    i64(pl.scan_parquet(os.path.join(WORK, "test_scores_full_cons.parquet")).select("q", "s").join(frs.lazy(), on="s").collect()),
    i64(pl.scan_parquet(os.path.join(WORK, "ce_x", "test_rows.parquet")).select("q", "s").join(frs.lazy(), on="s").collect()),
]).unique()
log(f"our France list pairs {lists.height}")

gin = g.join(lists, on=["q", "s"], how="semi")
log(f"gathik France pairs inside our lists {gin.height} ({gin.height / g.height:.3f})")
c = gin.join(vfr, on=["q", "s"], how="anti")
log(f"... not in v10b {c.height}")
vq = v.select("q", vs="s")
c = c.join(vq, on="q", how="left").with_columns(kind=pl.when(pl.col("vs").is_null()).then(pl.lit("add")).otherwise(pl.lit("move")))
log(str(c.group_by("kind").len()))

# veto sets
vetos = pl.concat([
    pl.read_parquet(os.path.join(TMP, "frfix", "namechg", "veto_set.parquet"), columns=["q", "s"]),
    pl.read_parquet(os.path.join(WORK, "frfix2", "fp_veto_set.parquet"), columns=["q", "s"]),
    pl.read_parquet(os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), columns=["q", "s"]),
    pl.read_parquet(os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet"), columns=["q", "s"]),
]).pipe(i64).unique()
nb = c.height
c = c.join(vetos, on=["q", "s"], how="anti")
log(f"after veto sets {c.height} (removed {nb - c.height})")

rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select("entity_id", "business_name", "business_address") for k in (2, 3)]
                ).select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address").join(c.select("q").unique().lazy(), on="q").collect()
c = c.join(fr, on="s").join(rec, on="q")

FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]


def forms(col):
    t = (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
         .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique()


c = c.with_columns(fs=forms("sn"), fq=forms("qn"))
conf = ((pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0) & (pl.col("fs").list.set_intersection("fq").list.len() == 0)).fill_null(False)
log(f"legal conflicts {c.filter(conf).height}")
c = c.filter(~conf).drop("fs", "fq")

# probabilities
ours = combo.rename({"p2": "po"})
c = c.join(ours, on=["q", "s"], how="left")
# our best competing S1 for this record and its prob
best = combo.sort("p2", descending=True).unique("q", keep="first").rename({"s": "s_best", "p2": "po_best"})
c = c.join(best, on="q", how="left")
gp = i64(pl.scan_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet")).select("q", "s", pg="p2").join(frs.lazy(), on="s").collect())
gx = i64(pl.scan_parquet(os.path.join(WORK, "gathik", "v9", "test_scores_full_xgb_cons.parquet")).select("q", "s", pgx="p2").join(frs.lazy(), on="s").collect())
c = c.join(gp, on=["q", "s"], how="left").join(gx, on=["q", "s"], how="left")
# prob of v10b's current S1 for moved records
c = c.join(ours.rename({"s": "vs", "po": "po_vs"}), on=["q", "vs"], how="left").join(gp.rename({"s": "vs", "pg": "pg_vs"}), on=["q", "vs"], how="left")
# S1 row state in v10b
nmatch = vfr.group_by("s").agg(n_v=pl.len())
c = c.join(nmatch, on="s", how="left").with_columns(pl.col("n_v").fill_null(0))
c = c.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or ""))) for r in c.iter_rows(named=True)], dtype=pl.Utf8),
                   low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
c = c.with_columns(nm=pl.col("ops").str.extract_all(r"n_[a-z_]+(?::[a-z]+)?").list.join("+"), num=pl.col("ops").str.extract(r"(a_num[a-z0-9_]*)"))
c.write_parquet(os.path.join(T, "cand.parquet"))
log(f"wrote cand {c.height}")
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(50)
log(str(c.group_by("kind").agg(n=pl.len(), low=pl.col("low").mean(), po=pl.col("po").mean(), pg=pl.col("pg").mean(), empty=(pl.col("n_v") == 0).mean())))
