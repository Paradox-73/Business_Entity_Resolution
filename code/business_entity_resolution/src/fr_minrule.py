"""France rows: the transformer may only LOWER a pair's probability (p2 = min(GBDT p2, stage-3 p3)).

Label-free audit of v7b vs v7b_frbase (26 Sep 05:20): France records the transformer REMOVED look like the known
France look-alikes (house number differs 20%, name word differs 73%), while records it ADDED differ in house number
34% of the time, vs 1.6% for records both keep. So removals look right and additions mostly wrong in France (the
transformer never saw France labels). US/India keep the stage-3 probability (validated on held-out labels).

  python fr_minrule.py <reranked scores.parquet> <GBDT scores.parquet> <out.parquet>
  e.g. fr_minrule.py work/ce/test_scores_ce_ab.parquet work/test_scores_full_cons.parquet work/ce/test_scores_ce_ab_frmin.parquet
"""
import os
import sys
import polars as pl
from common import WORK, log, id_to_int


def main(ce_path, gb_path, out):
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), country="country")
    ce = pl.read_parquet(ce_path, columns=["q", "s", "p1", "p2"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    gb = pl.read_parquet(gb_path, columns=["q", "s", "p2"]).rename({"p2": "p2g"}).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    b = ce.join(gb, on=["q", "s"], how="left").join(s1, on="s", how="left")
    fr = pl.col("country") == "France"
    # a France pair the GBDT never scored (stage-1 rank 3-5 extras) counts as GBDT 0: the transformer may not add it
    new = pl.when(fr).then(pl.min_horizontal("p2", pl.col("p2g").fill_null(0.0))).otherwise(pl.col("p2"))
    b = b.with_columns(p2n=new.cast(pl.Float32))
    ch = b.filter(fr & (pl.col("p2n") != pl.col("p2")))
    log(f"France pairs lowered back to the GBDT probability: {ch.height} "
        f"({ch.filter(pl.col('p2') >= 0.5).height} had stage-3 p >= 0.5, now {ch.filter(pl.col('p2n') >= 0.5).height})")
    b.select("q", "s", "p1", p2="p2n").write_parquet(out)
    log(f"wrote {out} ({b.height} rows)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
