"""Probabilities of v10a for the countries with training labels (US, India; common.labelled_countries): our pipeline
blended with the second pipeline, then the list-mover overrides.

  python blend_second.py <out_dir> [<movers_dir> [<reference_dir>]]
  v10a: python blend_second.py $BER_WORK/blend9 $BER_WORK/movers

For every (record q, S1 row s) pair of those countries that either pipeline scored:
  po = our probability, p2 of WORK/test_scores_blend_v7p.parquet (stage 3 of e5-small 0.3 + bge-reranker-v2-m3 0.7 on
       the wide US/India candidate lists; README step 15)
  pg = the second pipeline's probability, p2 of WORK/gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet (its stage
       3 with 3 bge-reranker-v2-m3 folds; README step 17)
  p2 = 0.6 po + 0.4 pg where both scored the pair, otherwise the one that scored it. The weight 0.4 was chosen on
       held-out (submissions/v10a/finalize.json); held-out macro F0.5 with all close calls rescored (EXPERIMENTS.md):
       ours alone 0.98950, second pipeline alone 0.99183, this blend 0.99205.
List-mover overrides (movers/build_movers.py): pairs in tier12_removed.parquet get p2 = 0; pairs in
tier12_restored.parquet get p2 = max(p2, 0.95), and restored pairs that neither pipeline scored enter with 0.95.
Decision: pipeline.decide_expf(p2, floor 0.5, alpha 1.0).

Writes to <out_dir>:
  test_scores_usi_blend_w04_movers.parquet  q, s, p2 for every US/India pair above (input of stack/build_test.py)
  cand_union.parquet                        q, s: every pair of test_scores_blend_v7p.parquet (all countries) plus
                                            the US/India pairs above (the candidate list of v10a)
  usi_pairs.parquet                         s, q: the decided US/India matches of v10a
With <reference_dir>, the first two files are compared with the files of the same name there.
"""
import os
import sys

import polars as pl

from common import WORK, id_to_int, is_labelled, log
from pipeline import decide_expf

W_SECOND = 0.4          # weight of the second pipeline where both pipelines scored a pair
RESTORE_P = 0.95        # probability floor of a restored list-mover pair


def i64(df):
    """q and s cast to Int64 (the score files use different integer types)."""
    return df.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def blend(movers_dir):
    """p2 per pair of the countries with training labels: 0.6 ours + 0.4 the second pipeline's where both scored it,
    then the list-mover overrides from movers_dir. Returns (scored pairs q, s, p2; candidate pairs q, s = cand_union)."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), country="country")
    usi = s1.filter(is_labelled()).select("s")
    ours_all = i64(pl.read_parquet(os.path.join(WORK, "test_scores_blend_v7p.parquet"), columns=["q", "s", "p2"]))
    po = ours_all.join(usi, on="s").rename({"p2": "po"})
    pg = i64(pl.read_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_ce_bgef0bgef1bgef2.parquet"),
                             columns=["q", "s", "p2"])).join(usi, on="s").rename({"p2": "pg"})
    d = po.join(pg, on=["q", "s"], how="full", coalesce=True)
    log(f"pairs of the countries with training labels: ours {po.height}, second pipeline {pg.height}, union {d.height}")
    d = d.select("q", "s", p2=pl.when(pl.col("po").is_null()).then(pl.col("pg"))
                 .when(pl.col("pg").is_null()).then(pl.col("po"))
                 .otherwise(W_SECOND * pl.col("pg") + (1 - W_SECOND) * pl.col("po")))
    rem = pl.read_parquet(os.path.join(movers_dir, "tier12_removed.parquet"), columns=["q", "s"]).with_columns(rm=pl.lit(1))
    res = pl.read_parquet(os.path.join(movers_dir, "tier12_restored.parquet"), columns=["q", "s"]).with_columns(rs=pl.lit(1))
    log(f"list movers: removed {rem.height}, restored {res.height} "
        f"({res.join(d, on=['q', 's'], how='anti').height} of them scored by neither pipeline)")
    d = (d.join(res.select("q", "s"), on=["q", "s"], how="full", coalesce=True)
          .join(rem, on=["q", "s"], how="left").join(res, on=["q", "s"], how="left")
          .with_columns(p2=pl.when(pl.col("rm") == 1).then(0.0)
                        .when(pl.col("rs") == 1).then(pl.max_horizontal(pl.col("p2").fill_null(0.0), pl.lit(RESTORE_P)))
                        .otherwise(pl.col("p2")).cast(pl.Float32))
          .select("q", "s", "p2").sort("q", "s"))
    cand = pl.concat([ours_all.select("q", "s"), d.select("q", "s")]).unique().sort("q", "s")
    return d, cand


def compare(name, new, ref_dir):
    """Log how `new` differs from the file `name` in ref_dir: pairs found in only one of them, and the
    largest p2 difference."""
    ref = i64(pl.read_parquet(os.path.join(ref_dir, name)))
    keys = ["q", "s"]
    only_new = new.join(ref, on=keys, how="anti").height
    only_ref = ref.join(new, on=keys, how="anti").height
    msg = f"{name}: rows {new.height} vs reference {ref.height}; only here {only_new}, only in reference {only_ref}"
    if "p2" in new.columns:
        j = new.join(ref.rename({"p2": "p2_ref"}), on=keys)
        msg += f"; max abs p2 difference {(j['p2'] - j['p2_ref']).abs().max()}"
    log(msg)


if __name__ == "__main__":
    out = sys.argv[1]
    movers = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "movers")
    os.makedirs(out, exist_ok=True)
    d, cand = blend(movers)
    d.write_parquet(os.path.join(out, "test_scores_usi_blend_w04_movers.parquet"))
    cand.write_parquet(os.path.join(out, "cand_union.parquet"))
    pairs = decide_expf(d, "p2", 0.5, 1.0)
    pairs.write_parquet(os.path.join(out, "usi_pairs.parquet"))
    log(f"wrote {d.height} scored US/India pairs, {cand.height} candidate pairs, {pairs.height} matched US/India pairs")
    if len(sys.argv) > 3:
        compare("test_scores_usi_blend_w04_movers.parquet", d, sys.argv[3])
        compare("cand_union.parquet", cand, sys.argv[3])
