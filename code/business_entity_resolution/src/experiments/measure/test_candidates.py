"""Candidate-set size on TEST, stage by stage, and the final candidate_pairs.tsv (per country).

Methodology section 3 ("What candidate_pairs.tsv contains", "Pairs at each stage", "Final candidate file by
country"); root README "Candidate generation".
Reads BER_WORK and the candidate file; writes $BER_SCRATCH/measure/test_candidates.json (default C:/ber_scratch).
  python test_candidates.py [candidate_pairs.tsv]     (default: <root>/submissions/v10d/candidate_pairs.tsv; in the zip
                                                        pass output/candidate_pairs.tsv; its matching_results.tsv must
                                                        sit in the same folder)
"""
import glob
import json
import os
import sys
import pyarrow.parquet as pq
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
from common import WORK, ROOT, id_to_int, log

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
os.makedirs(SCR, exist_ok=True)
CAND = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submissions", "v10d", "candidate_pairs.tsv")
MATCH = os.path.join(os.path.dirname(CAND), "matching_results.tsv")
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
C3 = ["France", "India", "US"]

s1 = pl.read_parquet(os.path.join(WORK, "pairs", "test", "s1.parquet")).select(pl.col("s").cast(pl.Int64), "country")
qt = pl.read_parquet(os.path.join(WORK, "pairs", "test", "q.parquet")).select(pl.col("q").cast(pl.Int64), "country", "addr_missing")
n_s1 = dict(s1.group_by("country").len().iter_rows())
n_q = dict(qt.group_by("country").len().iter_rows())
n_all = sum(n_s1[c] * n_q[c] for c in C3)
res = {"n_s1": n_s1, "n_records": n_q, "same_country_s1_x_record": {c: n_s1[c] * n_q[c] for c in C3}, "same_country_total": n_all,
       "all_s1_x_all_records": s1.height * qt.height}
log(res)


def per(pairs_by_c):
    """pairs per country -> pairs, per S1 row, per record, reduction ratio against same-country S1 x records."""
    out = {}
    for c in C3 + ["all"]:
        n = pairs_by_c.get(c, 0) if c != "all" else sum(pairs_by_c.get(k, 0) for k in C3)
        ns = n_s1[c] if c != "all" else s1.height
        nq = n_q[c] if c != "all" else qt.height
        full = n_s1[c] * n_q[c] if c != "all" else n_all
        out[c] = {"pairs": n, "per_s1": round(n / ns, 3), "per_record": round(n / nq, 3), "reduction_ratio": 1 - n / full}
    return out


# ---- 1. raw search output: row counts of the pair-feature chunk files (parquet metadata only)
for tag in ("test", "test_b2"):
    cnt = {}
    for f in glob.glob(os.path.join(WORK, "pairs", tag, "*_[0-9][0-9][0-9].parquet")):
        c = os.path.basename(f).split("_")[0]
        cnt[c] = cnt.get(c, 0) + pq.ParquetFile(f).metadata.num_rows
    res[f"raw_search_{tag}"] = per(cnt)
    log(tag, res[f"raw_search_{tag}"])


def by_country(path, label):
    d = i64(pl.read_parquet(path, columns=["q", "s"])).unique().join(s1, on="s")
    cnt = dict(d.group_by("country").len().iter_rows())
    r = per(cnt)
    for c in C3:
        dc = d.filter(pl.col("country") == c)
        r[c]["records_with_pairs"] = dc["q"].n_unique()
        r[c]["s1_with_pairs"] = dc["s"].n_unique()
    res[label] = r
    log(label, r)
    return d


# ---- 2. stage-2 input (each record's stage-1 top 2; 2nd only if p1 >= 0.01)
prod = by_country(os.path.join(WORK, "test_scores_full_cons.parquet"), "stage2_input_production")
wide = by_country(os.path.join(WORK, "test_scores_full_cons_test_b2.parquet"), "stage2_input_wide")
gat = by_country(os.path.join(WORK, "gathik", "v9", "test_scores_full_xgb_cons.parquet"), "stage2_input_second_pipeline")

# ---- 3. close calls rescored by the transformers (top 2 by p2 of uncertain records) + stage-1 ranks 3-5 of them
for path, label in ((os.path.join(WORK, "ce_x", "test_rows.parquet"), "transformer_rows_production"),
                    (os.path.join(WORK, "ce_b2", "test_rows.parquet"), "transformer_rows_wide"),
                    (os.path.join(WORK, "gathik", "v9", "ce_x_test_rows.parquet"), "transformer_rows_second_pipeline")):
    d = i64(pl.read_parquet(path, columns=["q", "s", "p2"])).join(s1, on="s")
    r = {}
    for c in C3 + ["all"]:
        dc = d if c == "all" else d.filter(pl.col("country") == c)
        r[c] = {"pairs": dc.height, "close_call_pairs": dc.filter(pl.col("p2").is_not_null()).height,
                "rank3_5_pairs": dc.filter(pl.col("p2").is_null()).height, "records": dc["q"].n_unique(),
                "share_of_records": round(dc["q"].n_unique() / (qt.height if c == "all" else n_q[c]), 4)}
    res[label] = r
    log(label, r)

# ---- 4. final candidate file
cf = pl.read_csv(CAND, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "candidate_entity_ids": pl.Utf8})
cf = cf.with_columns(s=id_to_int("source1_entity_id").cast(pl.Int64), k=pl.col("candidate_entity_ids").fill_null("").str.split(","))
cf = cf.with_columns(n=pl.col("k").list.eval(pl.element().filter(pl.element() != "")).list.len()).join(s1, on="s", how="left")
C = (cf.select("s", "k").explode("k").filter(pl.col("k").fill_null("") != "")
       .select("s", q=id_to_int("k").cast(pl.Int64)).join(s1, on="s"))
fin = per(dict(C.group_by("country").len().iter_rows()))
for c in C3 + ["all"]:
    x = cf if c == "all" else cf.filter(pl.col("country") == c)
    fin[c].update({"s1_rows": x.height, "median_per_s1": float(x["n"].median()), "p90_per_s1": float(x["n"].quantile(0.9)),
                   "max_per_s1": int(x["n"].max()), "s1_without_candidates": int((x["n"] == 0).sum())})
    cq = C if c == "all" else C.filter(pl.col("country") == c)
    nrec = cq["q"].n_unique()
    fin[c]["records_in_candidates"] = nrec
    fin[c]["share_records_in_candidates"] = round(nrec / (qt.height if c == "all" else n_q[c]), 4)
    fin[c]["s1_per_record_mean"] = round(cq.height / nrec, 3)
res["final_candidates"] = fin
res["final_candidates_file"] = os.path.relpath(CAND, ROOT)
log("final", fin)

# matched pairs of the same submission (candidates per matched pair)
m = pl.read_csv(MATCH, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
M = (m.with_columns(s=id_to_int("source1_entity_id").cast(pl.Int64), k=pl.col("matched_entity_ids").fill_null("").str.split(","))
       .select("s", "k").explode("k").filter(pl.col("k").fill_null("") != "").select("s", q=id_to_int("k").cast(pl.Int64)).join(s1, on="s"))
mc = dict(M.group_by("country").len().iter_rows())
res["matched_pairs"] = {c: mc.get(c, 0) for c in C3} | {"all": M.height, "not_in_candidates": M.join(C, on=["s", "q"], how="anti").height}

# ---- 5. where the final candidates come from (overlap with each generator's lists)
C = C.select("q", "s", "country")
v7p = i64(pl.read_parquet(os.path.join(WORK, "test_scores_blend_v7p.parquet"), columns=["q", "s"])).unique()
cu = i64(pl.read_parquet(os.path.join(WORK, "blend9", "cand_union.parquet"))).unique()
src = {"our_final_score_table(test_scores_blend_v7p)": v7p, "stage2_input_production": prod.select("q", "s"),
       "stage2_input_wide": wide.select("q", "s"), "stage2_input_second_pipeline": gat.select("q", "s"),
       "cand_union(blend9)": cu}
comp = {}
for k, d in src.items():
    j = C.join(d, on=["q", "s"])
    comp[k] = {"pairs_in_source": d.height, "final_candidates_in_source": dict(j.group_by("country").len().iter_rows()) | {"all": j.height}}
ours = pl.concat([v7p, gat.select("q", "s")]).unique()
extra = C.join(ours, on=["q", "s"], how="anti")
comp["final_not_in_ours_or_second_stage2"] = dict(extra.group_by("country").len().iter_rows()) | {"all": extra.height}
comp["cand_union_minus_(v7p_table+second_stage2)"] = cu.join(ours, on=["q", "s"], how="anti").height
comp["(v7p_table+second_stage2)_minus_cand_union"] = ours.join(cu, on=["q", "s"], how="anti").height
only_g = C.join(v7p, on=["q", "s"], how="anti").join(gat.select("q", "s"), on=["q", "s"])
comp["final_from_second_pipeline_only"] = dict(only_g.group_by("country").len().iter_rows()) | {"all": only_g.height}
res["composition"] = comp
log("composition", comp)

json.dump(res, open(os.path.join(SCR, "test_candidates.json"), "w"), indent=1)
log("wrote", os.path.join(SCR, "test_candidates.json"))
