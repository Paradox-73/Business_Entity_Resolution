"""Write matching_results.tsv from saved test scores with the calibrated per-country decision (calib.py).

  python finalize.py <model_dir_name> <out_dir> <scores.parquet>[:<countries>] [...]
  e.g. finalize.py tlike_xgb_cons out_v6 E:/.../test_scores_tlike_xgb_cons.parquet:US,India E:/.../test_scores_tlike_xgb_cons_testfr.parquet:France

Each scores file has q, s, p1, p2 (pipeline.predict writes WORK/test_scores_<model>[_<test_tag>].parquet).
Several files are concatenated (e.g. US/India from the full test build + France from the 'testfr' rebuild);
rows of a country must come from exactly one file. Countries without their own rule (France) use 'default'.
"""
import json
import os
import sys
import numpy as np
import polars as pl
from common import WORK, log, id_to_int, int_to_id
from pipeline import decide_expf

md, out_dir, files = os.path.join(WORK, "models", sys.argv[1]), sys.argv[2], sys.argv[3:]
# BER_CALIB=<file in the model dir> picks another rule file, e.g. raw_rule.json = uncalibrated p2 (v3's rule)
cal = json.load(open(os.path.join(md, os.environ.get("BER_CALIB", "calib.json"))))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
    s=id_to_int("entity_id"), s1_id="entity_id", country="country")
def load_scores(spec):
    """'path' (all countries) or 'path:US,India' (only these countries' rows)."""
    path, sep, cs = spec.rpartition(":")
    if not sep or cs.endswith(".parquet") or "/" in cs or "\\" in cs:     # no country list (colon of "E:/...")
        path, cs = spec, ""
    b = pl.read_parquet(path, columns=["q", "s", "p2"]).join(s1.select("s", "country"), on="s")
    return b.filter(pl.col("country").is_in(cs.split(","))) if cs else b


B = pl.concat([load_scores(f) for f in files])
dup = B.group_by("q").agg(pl.col("country").n_unique().alias("k")).filter(pl.col("k") > 1).height
assert dup == 0, f"{dup} records have candidates in two countries"
nf = B.select(pl.struct("q", "s").is_duplicated().sum()).item()
assert nf == 0, f"{nf} duplicated (q, s) rows: a country appears in two score files"
if cal.get("x"):
    B = B.with_columns(pc=pl.Series(np.interp(B["p2"].to_numpy(), cal["x"], cal["y"]).astype(np.float32)))
parts = []
for c in s1["country"].unique().sort().to_list():
    rule = cal["decision"].get(c, cal["decision"]["default"])
    bc = B.filter(pl.col("country") == c)
    if rule["type"] == "thr":
        m = bc.sort(rule["prob"], descending=True).unique("q", keep="first").filter(pl.col(rule["prob"]) >= rule["t"]).select("s", "q")
    else:
        m = decide_expf(bc, rule["prob"], rule["floor"], rule["alpha"])
    log(f"{c}: rule {rule}, {bc['q'].n_unique()} records scored, {m.height} matched, {m['s'].n_unique()} S1 non-empty")
    parts.append(m)
M = pl.concat(parts)
os.makedirs(out_dir, exist_ok=True)
df = M.group_by("s").agg(pl.col("q").sort()).with_columns(
    matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
out = (s1.join(df.select("s", "matched_entity_ids"), on="s", how="left")
         .with_columns(pl.col("matched_entity_ids").fill_null("")).select(source1_entity_id="s1_id", matched_entity_ids="matched_entity_ids"))
out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
json.dump({"model": sys.argv[1], "scores": files, "calib": cal["decision"], "oof_base": cal["base"], "oof_best": cal["best"]},
          open(os.path.join(out_dir, "result.json"), "w"), indent=1)
log(f"wrote {out_dir}/matching_results.tsv: {out.height} rows, {(out['matched_entity_ids'] != '').sum()} non-empty")
