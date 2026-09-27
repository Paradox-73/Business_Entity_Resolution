import sys, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/fp"))
from pipeline import decide_expf
from fp1_dec import france_scores
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
CUR = f"{WORK}/frfix2/test_scores_v9b_fpveto.parquet"
NEW = f"{WORK}/frfix3/test_scores_v9y_moreveto_stem.parquet"
acc = pl.read_parquet(f"{SCRATCH}/frfix3/moreveto/acc_v9y.parquet", columns=["s", "q"])
V = pl.read_parquet(f"{WORK}/frfix3/moreveto_stem_set.parquet")
ma = decide_expf(france_scores(CUR), "p2", 0.5, 1.0)
mb = decide_expf(france_scores(NEW), "p2", 0.5, 1.0)
print("v9y", acc.height, "base decide", ma.height, "new decide", mb.height)
print("base vs v9y: extra", ma.join(acc, on=["s","q"], how="anti").height, "missing", acc.join(ma, on=["s","q"], how="anti").height)
print(ma.join(acc, on=["s","q"], how="anti"))
rem = ma.join(mb, on=["s","q"], how="anti"); add = mb.join(ma, on=["s","q"], how="anti")
print("new vs base: removed", rem.height, "in veto", rem.join(V.select("s","q"), on=["s","q"]).height, "added", add.height)
print(add)
# k distribution of vetoed S1 rows in v9y
k = acc.group_by("s").len("k")
VV = V.join(k, on="s", how="left")
print("k dist", VV["k"].value_counts().sort("k").to_dicts())
print(V["npat"].value_counts())
# the ndiff one
t = pl.read_parquet(f"{SCRATCH}/france/fr_top.parquet", columns=["q","s","qa","sa","s_2","p2_2","sn_2"])
print(V.join(t, on=["q","s"]).filter(pl.col("npat")!="nsame").select("sn","qn","qa","sa","p2"))
x = V.join(t, on=["q","s"])
print("2nd candidate p2_2 >= 0.5:", (x["p2_2"] >= 0.5).sum(), x.select(pl.col("p2_2").max()))
