"""Implementation checks for fr_typo_in: ids exist, countries, record unmatched in v10b, not in v10b, not in veto sets, unique q."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int, read_tsv

R = os.path.dirname(WORK)
P = rf"{SCRATCH}/final/fr-gathik-rest/fr_typo_in.parquet"
x = pl.read_parquet(P)
print("pairs", x.height, "unique (s,q)", x.select("s", "q").unique().height, "unique q", x["q"].n_unique(), "unique s", x["s"].n_unique())
print("dtypes", x.schema["s"], x.schema["q"])
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), s_ctry="country",
                                                                  sn_raw="business_name", sa_raw="business_address").join(
    x.select("s").lazy(), on="s").collect()
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(q=id_to_int("entity_id").cast(pl.Int64), q_ctry="country",
                                                                                  qn_raw="business_name", qa_raw="business_address",
                                                                                  src=pl.lit(k)).join(x.select("q").lazy(), on="q").collect()
                 for k in (2, 3)])
print("S1 found", s1.height, "records found", rec.height, "record ids found in >1 source", rec.height - rec["q"].n_unique())
d = x.join(s1, on="s", how="left").join(rec, on="q", how="left")
print("S1 country", dict(d.group_by("s_ctry").len().iter_rows()), "record country", dict(d.group_by("q_ctry").len().iter_rows()))
print("country mismatch", d.filter(pl.col("s_ctry") != pl.col("q_ctry")).height)
print("name/addr in file == raw:", (d["sn"] == d["sn_raw"]).sum(), (d["qn"] == d["qn_raw"]).sum(), (d["sa"] == d["sa_raw"]).sum(), (d["qa"] == d["qa_raw"]).sum())


def pairs(p):
    t = read_tsv(p)
    return (t.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


v = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
print("v10b pairs", v.height, "records matched twice in v10b", v.height - v["q"].n_unique())
print("pairs already in v10b", x.join(v, on=["s", "q"]).height, "records matched (any S1) in v10b", x.join(v.select("q"), on="q").height)
cnt = v.group_by("s").len("n")
e = x.join(cnt, on="s", how="left").with_columns(pl.col("n").fill_null(0))
print("n_v10b column agrees", (e["n"] == e["n_v10b"]).all(), "S1 rows empty in v10b", e.filter(pl.col("n") == 0)["s"].n_unique())
i64 = lambda t: t.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
for vp in [rf"{SCRATCH}/frfix/namechg/veto_set.parquet", os.path.join(WORK, "frfix2", "fp_veto_set.parquet"),
           os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet")]:
    vs = i64(pl.read_parquet(vp, columns=["q", "s"]))
    print(os.path.basename(vp), vs.height, "pair hits", x.join(vs, on=["s", "q"]).height, "record hits", x.join(vs.select("q").unique(), on="q").height)
# gathik matched these?
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv"))
print("pairs in gathik matched output", x.join(g, on=["s", "q"]).height)
gp = i64(pl.scan_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet")).select("q", "s", "p2").join(x.select("q").lazy(), on="q").collect())
gg = x.select("s", "q", p_file="p2").join(gp, on=["s", "q"], how="left")
print("gathik p recomputed == file p2:", (gg["p2"] - gg["p_file"]).abs().max(), "min p", gg["p2"].min())
# other gathik candidates for these records with high p
oth = gp.join(x.select("s", "q"), on=["s", "q"], how="anti").filter(pl.col("p2") >= 0.3)
print("records with another gathik candidate p>=0.3:", oth["q"].n_unique())
# our lists
lists = pl.concat([i64(pl.read_parquet(os.path.join(WORK, "test_scores_full_cons.parquet"), columns=["q", "s"])),
                   i64(pl.read_parquet(os.path.join(WORK, "ce_x", "test_rows.parquet"), columns=["q", "s"])),
                   i64(pl.read_parquet(os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet"), columns=["q", "s"]))])
print("pairs in our lists", x.join(lists, on=["s", "q"]).height, "records in our lists (any S1)", x.join(lists.select("q").unique(), on="q").height)
d.select("s", "q", "s_ctry", "q_ctry", "src").write_parquet(rf"{SCRATCH}/final/verify/typo_impl.parquet")
