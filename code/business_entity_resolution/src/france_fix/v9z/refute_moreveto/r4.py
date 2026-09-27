import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
from common import WORK, id_to_int
CUR = f"{WORK}/frfix2/test_scores_v9b_fpveto.parquet"
NEW = f"{WORK}/frfix3/test_scores_v9y_moreveto_stem.parquet"
print(pl.read_parquet_schema(CUR)); print(pl.read_parquet_schema(NEW))
a = pl.read_parquet(CUR); b = pl.read_parquet(NEW)
print("rows", a.height, b.height, "cols", a.columns, b.columns)
# same order?
same_qs = (a["q"] == b["q"]).all() and (a["s"] == b["s"]).all()
print("same q,s order:", same_qs)
if not same_qs:
    b = b.join(a.select("q", "s").with_row_index("i"), on=["q", "s"], how="inner").sort("i").drop("i")
    print("after align rows", b.height)
for c in a.columns:
    if c in ("q", "s"): continue
    x, y = a[c], b[c]
    d = ~((x == y) | (x.is_null() & y.is_null()) | (x.is_nan() & y.is_nan() if x.dtype in (pl.Float32, pl.Float64) else False)).fill_null(False)
    print(c, a[c].dtype, b[c].dtype, "differ:", int(d.sum()))
diff = pl.DataFrame({"q": a["q"], "s": a["s"], "p2a": a["p2"], "p2b": b["p2"]}).filter(pl.col("p2a") != pl.col("p2b"))
print(diff.describe())
s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"]).select(s=id_to_int("entity_id"), country="country")
dd = diff.join(s1, on="s", how="left")
print(dd["country"].value_counts())
V = pl.read_parquet(f"{WORK}/frfix3/moreveto_stem_set.parquet", columns=["q", "s"])
print("diff == veto set:", dd.select("q","s").sort("q","s").equals(V.sort("q","s")), "veto rows", V.height, "diff rows", dd.height)
print("dup (q,s) in NEW:", b.select("q","s").is_duplicated().sum())
