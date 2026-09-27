import polars as pl, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from france_cal import toks
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(300)
top = pl.read_parquet(f"{F}/fr_top.parquet", columns=["q", "s", "p2", "p2g", "qn", "sn", "qa", "sa", "pat", "acc", "s_2", "p2_2", "pat_2"])
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
top = top.join(b.select("q", "s", "p3", "pr"), on=["q", "s"], how="left")
top = top.with_columns(kq=pl.Series([" ".join(sorted(set(toks(x)))) for x in top["qn"].to_list()]),
                       ks=pl.Series([" ".join(sorted(set(toks(x)))) for x in top["sn"].to_list()]))
top = top.with_columns(m=pl.len().over("s", "kq"), nrec=pl.len().over("s"))
top.write_parquet(f"{T}/top_k.parquet")
top = top.with_columns(nk=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last(),
                       rest=(pl.col("pr") >= 0.5) & (pl.col("p2") < 0.5))
print(top.group_by("nk").agg(n=pl.len(), acc=pl.col("acc").mean(), m1=(pl.col("m") == 1).mean(), m2=(pl.col("m") == 2).mean(), m3p=(pl.col("m") >= 3).mean()).sort("n", descending=True))
sw = top.filter(pl.col("nk") == "swap")
print("SWAP records by name-cluster size m (records with same best S1 and same name key):")
print(sw.group_by("m", "num").agg(n=pl.len(), acc=pl.col("acc").mean(), p2=pl.col("p2").mean(), g=pl.col("p2g").mean(), p3=pl.col("p3").mean(), rest=pl.col("rest").mean()).sort("num", "m").head(40))
