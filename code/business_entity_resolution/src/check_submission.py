"""Fast check of matching_results.tsv against every leaderboard rule (the official validator is slow on
the multi-GB candidate file). Usage: python check_submission.py <matching_results.tsv>"""
import sys
import polars as pl
from common import DATA

rd = lambda p, **k: pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0, **k)
m = rd(sys.argv[1])
assert m.columns == ["source1_entity_id", "matched_entity_ids"], m.columns
s1 = rd(f"{DATA}/test/test_source1.tsv", columns=["entity_id"])
q = pl.concat([rd(f"{DATA}/test/test_source{k}.tsv", columns=["entity_id"]) for k in (2, 3)])
e = (m.with_columns(ids=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("ids")
      .filter(pl.col("ids").fill_null("") != ""))
problems = {
    "duplicate S1 rows": m.height - m["source1_entity_id"].n_unique(),
    "missing S1 rows": s1.join(m, left_on="entity_id", right_on="source1_entity_id", how="anti").height,
    "unknown S1 rows": m.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height,
    "duplicate id in a list": e.height - e.unique(["source1_entity_id", "ids"]).height,
    "ids not in test S2/S3": e.join(q, left_on="ids", right_on="entity_id", how="anti").height,
}
print(f"{m.height} rows, {e.height} matched ids;", problems)
print("PASS" if not any(problems.values()) else "FAIL")
sys.exit(0 if not any(problems.values()) else 1)
