import polars as pl, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import read_truth, id_to_int
F = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
u = pl.read_parquet(f"{F}/usi_top.parquet", columns=["q", "s", "label", "p", "c", "pat"])
tr = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
ss = u.select("s").unique()
t = tr.join(ss, on="s")
print("usi_top S1:", ss.height, " true pairs of these S1:", t.height, " true pairs present in usi_top:", u["label"].sum(),
      " true records of these S1 present in usi_top (any s):", t.join(u.select("q"), on="q").height)
