import polars as pl, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from france_cal import toks
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(300)
top = pl.read_parquet(f"{T}/top_k.parquet").with_columns(nk=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last(),
       rest=(pl.col("pr") >= 0.5) & (pl.col("p2") < 0.5))
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet")
s1 = s1.with_columns(ks=pl.Series([" ".join(sorted(set(toks(x)))) for x in s1["sn"].to_list()]))
rc = pl.read_parquet(f"{T}/fr_recsk.parquet", columns=["q", "qcity"])
top = top.join(rc, on="q", how="left")
cnt_city = s1.group_by("ks", "scity").agg(nx_city=pl.len(), sx=pl.col("s"))
cnt_all = s1.group_by("ks").agg(nx_all=pl.len())
top = top.join(cnt_all, left_on="kq", right_on="ks", how="left").join(cnt_city, left_on=["kq", "qcity"], right_on=["ks", "scity"], how="left").with_columns(
    pl.col("nx_all").fill_null(0), pl.col("nx_city").fill_null(0))
top = top.with_columns(xin=pl.col("sx").list.contains(pl.col("s")).fill_null(False))
top.drop("sx").write_parquet(f"{T}/top_k2.parquet")
f = top.filter(pl.col("nk").is_in(["swap", "added", "other"]))
print("records with swap/added/other best S1: exact-name S1 elsewhere in France / in same city")
print(f.group_by("nk", "num").agg(n=pl.len(), acc=pl.col("acc").mean(), ex_all=(pl.col("nx_all") > 0).mean(), ex_city=(pl.col("nx_city") > 0).mean(),
      acc_ex_city=pl.col("acc").filter(pl.col("nx_city") > 0).mean(), acc_noex=pl.col("acc").filter(pl.col("nx_city") == 0).mean(),
      rest_ex=pl.col("rest").filter(pl.col("nx_city") > 0).sum(), rest_noex=pl.col("rest").filter(pl.col("nx_city") == 0).sum()).sort("n", descending=True))
