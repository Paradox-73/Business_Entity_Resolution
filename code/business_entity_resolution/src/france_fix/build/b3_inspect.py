import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, hashlib
A = f"{WORK}/frfix/test_scores_frmin_descveto.parquet"
B = f"{SCRATCH}/frfix/namechg/test_scores_frmin_descveto.parquet"
h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
print("same file:", h(A) == h(B))
print(pl.read_parquet_schema(B))
FR = f"{SCRATCH}/france"
t = pl.read_parquet(f"{FR}/fr_top.parquet"); print(t.columns, t.height)
u = pl.read_parquet(f"{FR}/usi_top.parquet"); print(u.columns, u.height)
