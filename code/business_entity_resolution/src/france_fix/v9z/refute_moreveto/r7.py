import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(220)
one = pl.read_parquet(f"{SCRATCH}/frfix3/refute_moreveto/one_swaps.parquet")
D = one.filter(pl.col("ca") == "D")
pairs = [("sport","sportive"),("sportive","sport"),("college","collectif"),("collectif","college"),("groupe","groupement")]
for d, a in pairs:
    nd = D.filter(pl.col("d") == d).height
    oth = D.filter((pl.col("d") != d) & (pl.col("d") != a))
    share = oth.filter(pl.col("a") == a).height / oth.height
    obs = D.filter((pl.col("d") == d) & (pl.col("a") == a)).height
    print(f"{d}->{a}: observed {obs}, expected if random D pick {nd*share:.1f} (D replacements from {d}: {nd}, share of {a} elsewhere {share:.4f})")
# k distribution comparison: accepted D->D same-number pairs in v7ens (acc) on their S1 rows, v7ens accepted count per S1
t = pl.read_parquet(f"{SCRATCH}/france/fr_top.parquet", columns=["q","s","acc"]).filter(pl.col("acc"))
k7 = t.group_by("s").len("k")
x = one.filter(pl.col("acc") & (pl.col("num")=="NSAME")).join(k7, on="s", how="left")
x = x.with_columns(g=pl.when((pl.col("d").is_in(["sport","sportive","college","collectif","groupe"])) & (pl.col("a").is_in(["sport","sportive","college","collectif","groupement"])) & (pl.col("a").str.slice(0,5)==pl.col("d").str.slice(0,5))).then(pl.lit("STEM"))
                   .when((pl.col("cd")=="D") & (pl.col("ca")=="D")).then(pl.lit("DD")).when(pl.col("ca")=="N").then(pl.lit("xN")).otherwise(pl.lit("other")))
print(x.group_by("g").agg(n=pl.len(), k1=(pl.col("k")==1).mean(), k2=(pl.col("k")==2).mean(), k3_5=pl.col("k").is_between(3,5).mean(), k6p=(pl.col("k")>=6).mean(), kmean=pl.col("k").mean()))
print("all v7ens accepted pairs k:", t.join(k7, on="s").select(k1=(pl.col("k")==1).mean(), k3_5=pl.col("k").is_between(3,5).mean(), kmean=pl.col("k").mean()))
