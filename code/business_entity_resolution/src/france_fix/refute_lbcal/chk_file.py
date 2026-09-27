import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import numpy as np, polars as pl, pyarrow.parquet as pq
from common import WORK, id_to_int
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(s=id_to_int("entity_id"), fr=pl.col("country") == "France")
frs = set(s1.filter("fr")["s"].to_list())
A = pq.ParquetFile(rf"{WORK}/ce/test_scores_blend_ab_a2_frmin.parquet")
B = pq.ParquetFile(rf"{SCRATCH}/frfix/lbcal/test_scores_frlbcal.parquet")
nd = 0; nd_nonfr = 0; nfr = 0; keymis = 0; p1d = 0; up = 0; down = 0
chg_s = []
for i in range(A.num_row_groups):
    a = pl.from_arrow(A.read_row_group(i)); b = pl.from_arrow(B.read_row_group(i))
    assert a.height == b.height
    keymis += int(((a["q"] != b["q"]) | (a["s"] != b["s"])).sum())
    p1d += int((a["p1"] != b["p1"]).sum())
    d = a["p2"] != b["p2"]
    isfr = a["s"].is_in(list(frs))
    nfr += int(isfr.sum())
    nd += int(d.sum()); nd_nonfr += int((d & ~isfr).sum())
    up += int((b["p2"] > a["p2"]).sum()); down += int((b["p2"] < a["p2"]).sum())
print("rows", A.metadata.num_rows, B.metadata.num_rows, "key mismatches", keymis, "p1 diffs", p1d)
print("France pair rows", nfr, "p2 changed", nd, "of which non-France", nd_nonfr, "up", up, "down", down)
