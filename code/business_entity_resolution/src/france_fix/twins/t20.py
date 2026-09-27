import polars as pl, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from common import read_truth, id_to_int
F = rf"{SCRATCH}/france"
u = pl.read_parquet(f"{F}/usi_top.parquet", columns=["q", "s", "label", "p", "c", "pat"])
tr = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
ss = u.select("s").unique()
t = tr.join(ss, on="s")
print("usi_top S1:", ss.height, " true pairs of these S1:", t.height, " true pairs present in usi_top:", u["label"].sum(),
      " true records of these S1 present in usi_top (any s):", t.join(u.select("q"), on="q").height)
