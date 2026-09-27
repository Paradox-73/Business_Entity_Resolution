import os, sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import WORK, id_to_int
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
F = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
# France S1 and France records independently
S = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(pl.col("country") == "France").select(s=id_to_int("entity_id")).collect()
Q = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).filter(pl.col("country") == "France").select(q=id_to_int("entity_id")).collect() for k in (2, 3)])
print("France S1", S.height, "France records", Q.height)
g = pl.scan_parquet(os.path.join(WORK, "test_scores_full_cons.parquet")).select("q", "s", "p2")
# candidates of France records, to any S1
gq = g.join(Q.lazy(), on="q").collect()
print("pairs of France records:", gq.height, " records with cand:", gq["q"].n_unique())
print("  of which S1 is France:", gq.join(S, on="s").height)
print("cands per France record:", gq.group_by("q").len()["len"].value_counts().sort("len"))
# France S1 pairs with non-France records
gs = g.join(S.lazy(), on="s").collect()
print("pairs of France S1:", gs.height, " with France record:", gs.join(Q, on="q").height)
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet")
print("v7ens accepted France pairs:", acc.height, " unique q:", acc["q"].n_unique())
