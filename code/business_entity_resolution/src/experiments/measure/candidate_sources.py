"""Where the final candidate pairs outside our score table come from, and the S1 rows with the most candidates
(methodology section 3: "What candidate_pairs.tsv contains", long lists).

  python candidate_sources.py        (from any folder; prints)

Reads BER_WORK, the France pair sets in sets/ (BER_SETS) and <root>/submissions/v10d/ (both final files).
"""
import os, sys
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
from common import WORK, ROOT, id_to_int, read_tsv  # noqa: E402

SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sets"))

i64 = lambda d: d.select(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).with_columns(
    s=id_to_int("entity_id").cast(pl.Int64))


def tsv_pairs(p, col):
    d = read_tsv(p)
    return (d.with_columns(k=pl.col(col).fill_null("").str.split(",")).explode("k").filter(pl.col("k") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("k").cast(pl.Int64)))


C = tsv_pairs(os.path.join(ROOT, "submissions", "v10d", "candidate_pairs.tsv"), "candidate_entity_ids")
M = tsv_pairs(os.path.join(ROOT, "submissions", "v10d", "matching_results.tsv"), "matched_entity_ids")
v7p = i64(pl.read_parquet(os.path.join(WORK, "test_scores_blend_v7p.parquet"), columns=["q", "s"]))
gat = i64(pl.read_parquet(os.path.join(WORK, "gathik", "v9", "test_scores_full_xgb_cons.parquet"), columns=["q", "s"]))
x = C.join(v7p, on=["q", "s"], how="anti").join(s1.select("s", "country"), on="s")
print("final candidates outside our score table:", dict(x.group_by("country").len().iter_rows()), "total", x.height)
print("  of which matched in v10d:", x.join(M, on=["q", "s"]).height)
print("  of which in the second pipeline's stage-2 input:", x.join(gat, on=["q", "s"]).height)
g8 = tsv_pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv"), "matched_entity_ids")
rest = x.join(gat, on=["q", "s"], how="anti")
print("  outside both tables:", dict(rest.group_by("country").len().iter_rows()), "; in the second pipeline's earlier (v8) matched file:",
      rest.join(g8, on=["q", "s"]).height)
for f in ("fr_same_address_safe", "fr_typo_safe", "fr_amp_safe"):
    p = pl.read_parquet(os.path.join(SETS, f + ".parquet"))
    p = i64(p)
    print(f"  {f}: {p.height} pairs, outside both tables {rest.join(p, on=['q','s']).height}, matched in v10d {p.join(M, on=['q','s']).height}")
print("  outside both tables and all of the above:", rest.join(g8, on=["q", "s"], how="anti").join(
    pl.concat([i64(pl.read_parquet(os.path.join(SETS, f + ".parquet"))) for f in ("fr_same_address_safe", "fr_typo_safe", "fr_amp_safe")]),
    on=["q", "s"], how="anti").height)
# the same-address rule's pairs in the final files (methodology section 3, "Third source")
sa = pl.read_parquet(os.path.join(SETS, "fr_same_address_safe.parquet"), columns=["q", "s", "acr"]).with_columns(
    pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
sam = sa.join(M, on=["q", "s"])
cu = i64(pl.read_parquet(os.path.join(WORK, "blend9", "cand_union.parquet"), columns=["q", "s"]))
sa_new = sam.join(cu, on=["q", "s"], how="anti")
print(f"fr_same_address_safe pairs matched in v10d: {sam.height}; in cand_union {sam.height - sa_new.height}; "
      f"else in the second pipeline's (v8) matched file {sa_new.join(g8, on=['q', 's']).height}; "
      f"new candidates {sa_new.join(g8, on=['q', 's'], how='anti').height} "
      f"(acronym names {sa_new.join(g8, on=['q', 's'], how='anti')['acr'].sum()})")
n = C.group_by("s").len("n").join(s1, on="s").sort("n", descending=True)
print("S1 rows with the most candidates:")
print(n.head(8).select("country", "n", "business_name", "business_address"))
print("S1 rows with > 50 candidates:", dict(n.filter(pl.col("n") > 50).group_by("country").len().iter_rows()),
      "; > 100:", n.filter(pl.col("n") > 100).height)

# matched pairs of v10d that only the added generators (wide search, second pipeline, same-address set) supplied
prod = pl.concat([i64(pl.read_parquet(os.path.join(WORK, "test_scores_full_cons.parquet"), columns=["q", "s"])),
                  i64(pl.read_parquet(os.path.join(WORK, "ce_x", "test_rows.parquet"), columns=["q", "s"]))]).unique()
Mc = M.join(s1.select("s", "country"), on="s")
only = Mc.join(prod, on=["q", "s"], how="anti")
print("v10d matched pairs by country:", dict(Mc.group_by("country").len().iter_rows()))
print("  not in the production lists (stage-2 input + ranks 3-5):", dict(only.group_by("country").len().iter_rows()), "total", only.height)
print("    of which in the wide lists (v7p table):", only.join(v7p, on=["q", "s"]).height,
      "; only in the second pipeline's stage-2 input:", only.join(v7p, on=["q", "s"], how="anti").join(gat, on=["q", "s"]).height)
