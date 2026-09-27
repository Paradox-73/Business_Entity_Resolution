import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
T = rf"{SCRATCH}/frfix/twins"
F = rf"{SCRATCH}/france"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(300)
t = pl.read_parquet(f"{T}/twin_addr.parquet")
rank = {"same": 0, "added": 1, "dropped": 1, "squashed": 2, "acronym": 2, "swap": 3, "other": 5, "empty": 5}
t = t.with_columns(r=pl.col("nk").replace_strict(rank))
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet")
top = pl.read_parquet(f"{F}/fr_top.parquet", columns=["q", "s", "pat", "s_2", "acc", "p2", "p2g"])
per = t.group_by("q").agg(nsame=(pl.col("nk") == "same").sum(), best=pl.col("r").min(), nswap=(pl.col("nk")=="swap").sum(), ns=pl.col("ns").first())
# accepted pair of the record (v7ens), wherever it is
a = acc.join(t.select("q", "s", "nk", "r"), on=["q", "s"], how="left")
x = per.join(a.rename({"s": "sa_"}), on="q", how="left")
print("twin-address records:", x.height, " accepted anywhere:", x["sa_"].is_not_null().sum(), " accepted to an S1 at the same address:", x["nk"].is_not_null().sum())
x = x.with_columns(state=pl.when(pl.col("sa_").is_null()).then(pl.lit("unacc")).when(pl.col("nk").is_null()).then(pl.lit("acc_elsewhere")).otherwise(pl.col("nk")))
print(x.group_by("state", "nsame").len().sort("state", "nsame"))
# wrong twin: accepted S1 at the address is not 'same' while a 'same' S1 exists at address
w = x.filter((pl.col("nsame") >= 1) & (pl.col("state") != "same") & (pl.col("state") != "unacc"))
print("accepted to a non-exact S1 while an exact-name S1 exists at the SAME address:", w.height)
print(w.group_by("state").len())
ws = w.select("q", sacc="sa_").join(t.filter(pl.col("nk") == "same").select("q", sx="s", gx="g", p3x="p3", pfx="pf", candx="cand"), on="q")
print("  exact S1 in candidate list:", ws["candx"].sum(), "of", ws.height)
print(ws.filter(pl.col("candx")).select("gx", "p3x", "pfx").describe())
ws.write_parquet(f"{T}/wrong_twin.parquet")
recs = pl.read_parquet(f"{T}/fr_recs.parquet"); s1 = pl.read_parquet(f"{T}/fr_s1.parquet")
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
smp = ws.sample(min(40, ws.height), seed=1).join(recs.select("q", "qn", "qa"), on="q").join(s1.select(sacc="s", an="sn", aa="sa"), on="sacc").join(s1.select(sx="s", xn="sn"), on="sx").join(b.select("q", sacc="s", ga="g", pfa="pf"), on=["q", "sacc"], how="left")
print(smp.select("qn", "an", "xn", "ga", "pfa", "gx", "pfx", "qa", "aa"))
# unaccepted records with an exact-name S1 at same address
u = x.filter((pl.col("state") == "unacc") & (pl.col("nsame") >= 1))
print("UNaccepted records with exact-name S1 at the same address:", u.height)
us = u.select("q").join(t.filter(pl.col("nk") == "same").select("q", "s", "g", "p3", "pf", "cand", "ns"), on="q")
print("  exact S1 in candidates:", us["cand"].sum(), " describe:", us.filter(pl.col("cand")).select("g", "p3", "pf").describe())
