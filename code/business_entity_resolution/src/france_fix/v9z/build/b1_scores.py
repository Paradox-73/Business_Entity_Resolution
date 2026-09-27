"""Build scores files: stem2 = moreveto_stem + 2 promoted pairs zeroed; combo = recall + all 73 zeros."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
import polars as pl
W = f"{WORK}/frfix3/"
print("polars", pl.__version__)
extra = pl.DataFrame({"q": [30565799217, 30369260852], "s": [10659305782, 10354232030]})
v71 = pl.read_parquet(W + "moreveto_stem_set.parquet", columns=["q", "s"])
v73 = pl.concat([v71, extra]).unique()
assert v73.height == 73
v73.write_parquet(W + "moreveto_stem2_set.parquet")

def zero(src, dst, pairs):
    b = pl.read_parquet(src)
    n0 = b.height
    b = b.with_row_index("i").join(pairs.with_columns(z=pl.lit(True)), on=["q", "s"], how="left").sort("i")
    hit = b["z"].fill_null(False)
    print(dst, "pairs to zero", pairs.height, "rows found", int(hit.sum()), "of which p2>0 before", int((hit & (b["p2"] > 0)).sum()))
    assert int(hit.sum()) == pairs.height
    b = b.with_columns(p2=pl.when(hit).then(0.0).otherwise(pl.col("p2")).cast(b.schema["p2"])).drop("i", "z")
    assert b.height == n0
    b.write_parquet(dst)
    return b

zero(W + "test_scores_v9y_moreveto_stem.parquet", W + "test_scores_v9y_stem2.parquet", extra)
zero(W + "test_scores_v9y_recall.parquet", W + "test_scores_v9y_combo.parquet", v73)

# sanity: compare to the base file
base = pl.read_parquet(f"{WORK}/frfix2/test_scores_v9b_fpveto.parquet")
for name in ["stem2", "combo", "recall"]:
    n = pl.read_parquet(W + f"test_scores_v9y_{name}.parquet")
    m = n.height
    same_order = (n.head(base.height)["q"] == base["q"]).all() and (n.head(base.height)["s"] == base["s"]).all()
    dp1 = int((n.head(base.height)["p1"] != base["p1"]).sum())
    dp2 = n.head(base.height).with_columns(b=base["p2"]).filter(pl.col("p2") != pl.col("b"))
    print(name, "rows", m, "extra rows", m - base.height, "same order", same_order, "p1 diffs", dp1,
          "p2 diffs", dp2.height, "set to 0:", dp2.filter(pl.col("p2") == 0).height,
          "raised:", dp2.filter(pl.col("p2") > pl.col("b")).height)
