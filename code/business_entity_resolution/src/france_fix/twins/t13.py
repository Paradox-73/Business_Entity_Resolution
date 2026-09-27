import polars as pl, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from france_cal import toks
from fr_restore import street
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(42); pl.Config.set_tbl_width_chars(330)
top = pl.read_parquet(f"{T}/top_k2.parquet")
u = pl.read_parquet(f"{T}/us_top_k.parquet")
print("US swap|nsame ex by nx_city:", u.filter((pl.col("pat") == "swap|nsame") & (pl.col("nx_city") > 0)).group_by(pl.col("nx_city").clip(0, 5)).agg(pl.len(), pl.col("label").mean(), pl.col("true_in_x").mean()).sort("nx_city"))
print("US all-pattern ex rate by nx_city (excluding same):", u.filter((pl.col("nk") != "same") & (pl.col("nx_city") > 0)).group_by(pl.col("nx_city").clip(0, 5)).agg(pl.len(), pl.col("label").mean(), pl.col("true_in_x").mean()).sort("nx_city"))
f = top.filter(pl.col("pat") == "swap|nsame").with_columns(bucket=pl.col("nx_city").clip(0, 4))
print("France swap|nsame by exact-name same-city S1 multiplicity:")
print(f.group_by("bucket").agg(n=pl.len(), acc=pl.col("acc").mean(), nacc=pl.col("acc").sum(), p2=pl.col("p2").mean(), g=pl.col("p2g").mean(), p3=pl.col("p3").mean(), rest=pl.col("rest").sum() if "rest" in f.columns else pl.len()).sort("bucket"))
# twin S1 details for bucket 1
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet")
s1 = s1.with_columns(ks=pl.Series([" ".join(sorted(set(toks(x)))) for x in s1["sn"].to_list()]))
rc = pl.read_parquet(f"{T}/fr_recsk.parquet", columns=["q", "qcity", "qnum", "qst"])
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet")
nacc = acc.group_by("s").agg(kacc=pl.len())
f1 = f.filter(pl.col("nx_city") == 1).join(rc, on="q").join(s1.select(sx="s", xn="sn", xa="sa", xnum="snum", xst="sst", ks="ks", scity="scity"), left_on=["kq", "qcity"], right_on=["ks", "scity"])
f1 = f1.join(nacc.rename({"s": "sx", "kacc": "kacc_x"}), on="sx", how="left").join(nacc, on="s", how="left").with_columns(pl.col("kacc_x").fill_null(0), pl.col("kacc").fill_null(0))
f1 = f1.with_columns(x_same_street=(pl.col("xst") == pl.col("qst")), x_same_num=(pl.col("xnum") == pl.col("qnum")), x_is_cand=(pl.col("sx") == pl.col("s_2")))
print("France swap|nsame with exactly one exact-name S1 in city:", f1.height)
print(f1.group_by("acc").agg(n=pl.len(), same_street=pl.col("x_same_street").mean(), same_num=pl.col("x_same_num").mean(), x_is_cand=pl.col("x_is_cand").mean(),
      kacc_x=pl.col("kacc_x").mean(), kacc=pl.col("kacc").mean(), kx0=(pl.col("kacc_x") == 0).mean()))
f1.write_parquet(f"{T}/swap_ex1.parquet")
print(f1.filter(pl.col("acc")).sample(20, seed=5).select("qn", "sn", "xn", "qa", "xa", "kacc", "kacc_x", "p2", "p3"))
