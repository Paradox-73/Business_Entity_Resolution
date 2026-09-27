"""candidate_pairs.tsv = the pairs the final matching models run inference over (problem statement: "the last
blocking/filtering stage"): for every test record, the stage-1 model's top candidates (stage-2 input, pipeline.topk)
plus the stage-1 ranks 3-5 of close-call records that the transformers score (rerank.py extras), i.e. every (q, s) of
the final stage-3 score file; plus, optionally, pairs from a second candidate generator (Gathik's v8 blocking) that the
final file matched. Checks that every matched pair of the submission is a candidate.

  python make_candidates.py <final_scores.parquet> <matching_results.tsv> <out_dir> [extra_pairs.tsv|.parquet ...]
e.g. python make_candidates.py work/test_scores_blend_v7p.parquet submissions/v9y/matching_results.tsv work/out_cand
                               work/gathik/v8_matching_results.tsv
"""
import os
import sys
import polars as pl
from common import WORK, id_to_int, int_to_id, log


def pairs(p):
    """(s, q) integer pairs of a pair-set parquet file, or of a matching_results.tsv / candidate_pairs.tsv."""
    if p.endswith(".parquet"):   # a pair set (s, q) from another generator, e.g. the same-address generator
        return pl.read_parquet(p, columns=["s", "q"]).with_columns(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64))
    d = pl.read_csv(p, separator="\t", schema_overrides={d_: pl.Utf8 for d_ in ("matched_entity_ids", "candidate_entity_ids")})
    col = d.columns[1]
    return (d.filter(pl.col(col).fill_null("") != "").select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=pl.col(col).str.split(","))
             .explode("q").with_columns(q=id_to_int("q").cast(pl.Int64)))


def main(scores, matching, out, *extra):
    """Write <out>/candidate_pairs.tsv: every (q, s) of the score table `scores`, plus, for each extra file, its
    pairs that `matching` matches and that are not yet candidates. Stops if a matched pair is not a candidate."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id"]).select(
        s1_id="entity_id", s=id_to_int("entity_id").cast(pl.Int64))
    C = pl.read_parquet(scores, columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    M = pairs(matching)
    for e in extra:        # a second generator's pairs: only those the final file actually matched
        add = pairs(e).join(M, on=["s", "q"]).join(C, on=["s", "q"], how="anti")
        log(f"{e}: {add.height} matched pairs from this generator added as candidates")
        C = pl.concat([C, add.select("q", "s")])
    C = C.unique()
    miss = M.join(C, on=["s", "q"], how="anti")
    log(f"candidates {C.height} pairs ({C.height / s1.height:.2f} per S1 row); matched {M.height}, not in candidates {miss.height}")
    assert miss.height == 0, "matched pairs missing from the candidate set"
    df = C.group_by("s").agg(pl.col("q").sort()).with_columns(candidate_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    o = (s1.join(df.select("s", "candidate_entity_ids"), on="s", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
           .select(source1_entity_id="s1_id", candidate_entity_ids="candidate_entity_ids"))
    os.makedirs(out, exist_ok=True)
    o.write_csv(os.path.join(out, "candidate_pairs.tsv"), separator="\t", quote_style="never")
    log(f"wrote {out}/candidate_pairs.tsv: {o.height} rows, {(o['candidate_entity_ids'] != '').sum()} non-empty")


if __name__ == "__main__":
    main(*sys.argv[1:])
