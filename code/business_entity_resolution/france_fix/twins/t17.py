import polars as pl, os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import WORK, id_to_int
from fr_restore import street
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(330)
u = pl.read_parquet(f"{T}/us_top_k.parquet").filter((pl.col("nx_city") > 0) & (pl.col("nk") != "same"))
ts = pl.scan_parquet(os.path.join(WORK, "train_s1.parquet")).select(ts=id_to_int("entity_id"), tn="business_name", ta="business_address").join(u.select("ts").drop_nulls().lazy(), on="ts").collect()
u = u.join(ts, on="ts", how="left")
x = u.filter(pl.col("true_in_x"))
nq = [street(a)[0] for a in x["qa"].to_list()]; nt = [street(a)[0] for a in x["ta"].to_list()]
print("US records true to exact-name twin X (not best):", x.height, " record number == X number:", sum(a == b and a is not None for a, b in zip(nq, nt)) / x.height)
print(x.sample(12, seed=1).select("pat", "qn", "sn", "tn", "qa", "sa", "ta", "p"))
