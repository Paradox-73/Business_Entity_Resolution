"""US/India: keep the old candidate build's scores, except for records whose best S1 under the wide build
(blocking_b2.py) is an S1 the old build never listed for them. Those records take all their rows from the wide build.

Why (27 Sep 07:20): on test, the wide build gives 12,488 US/India records a newly found best S1 with GBDT p2 >= 0.5
(labelled sample, stage 1: such additions are 87-100% right at p1 >= 0.9). It also moves records whose best S1 is
unchanged: 49,573 fall below p2 0.5 and 17,343 rise above it, because stage-2 group features (records per S1,
candidates per record) shift with ~50 instead of ~31 candidates per record. On the labelled sample those stage-1
movers are mostly false either way (27-32% true), but stage 2 was never validated on wide lists. This file takes only
the additions; blocking_b2's full output is the other variant.

  python hybrid_b2.py <old_scores> <wide_scores> <out> [old_list=WORK/test_scores_full_cons.parquet]
Scores files: q, s, p1, p2 over all test pairs (e.g. WORK/test_scores_blend_v7f.parquet and
WORK/ce_b2/test_scores_ce_smallfolds.parquet). France rows are always the old file's.
"""
import os
import sys
import polars as pl
from common import WORK, log, id_to_int


def main(old_path, wide_path, out, old_list=os.path.join(WORK, "test_scores_full_cons.parquet")):
    cols = ["q", "s", "p1", "p2"]
    old = pl.read_parquet(old_path, columns=cols).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    wide = pl.read_parquet(wide_path, columns=cols).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    listed = pl.read_parquet(old_list, columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    fr = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).filter(
        pl.col("country") == "France").select(s=id_to_int("entity_id").cast(pl.Int64))
    best = wide.sort("p2", descending=True).unique("q", keep="first").join(fr, on="s", how="anti")
    moved = best.join(listed, on=["q", "s"], how="anti").select("q")
    hi = best.join(moved, on="q").filter(pl.col("p2") >= 0.5).height
    b = pl.concat([old.join(moved, on="q", how="anti"),
                   wide.join(moved, on="q").with_columns(pl.col("p1").cast(old["p1"].dtype), pl.col("p2").cast(old["p2"].dtype))])
    log(f"records whose wide best S1 is new: {moved.height} ({hi} with p2 >= 0.5); rows old {old.height} -> {b.height}")
    b.write_parquet(out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
