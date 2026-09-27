import polars as pl, sys, re, unicodedata
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
F = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(300)
def norm(col):
    return (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
s1 = pl.read_parquet(f"{T}/fr_s1.parquet").with_columns(sk=norm("sn").str.split(" ").list.sort().list.join(" "))
recs = pl.read_parquet(f"{T}/fr_recs.parquet").with_columns(qk=norm("qn").str.split(" ").list.sort().list.join(" "))
print("S1 distinct full-name keys", s1["sk"].n_unique(), "of", s1.height)
dup = s1.group_by("sk").len()
print("S1 name-key multiplicity:", dup["len"].value_counts().sort("len").head(8))
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet")
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
# for each record: exact-name S1 set
ex = recs.join(s1.select("s", "sk"), left_on="qk", right_on="sk", how="inner").select("q", sx="s")
print("records with >=1 exact-name S1 anywhere in France:", ex["q"].n_unique(), "of", recs.height)
a = acc.join(ex, on="q", how="left")
print("accepted records:", acc.height)
x = a.group_by("q").agg(s=pl.col("s").first(), has=pl.col("sx").is_not_null().any(), same=(pl.col("sx") == pl.col("s")).any())
print("accepted with exact-name S1 existing:", x["has"].sum(), " of which accepted S1 IS exact:", x["same"].sum())
bad = x.filter(pl.col("has") & ~pl.col("same"))
print("accepted to non-exact S1 while an exact-name S1 exists:", bad.height)
# is the exact S1 in the candidate list?
bx = bad.select("q", "s").join(ex, on="q").join(b.rename({"s": "sx"}), on=["q", "sx"], how="left")
print("  exact S1 in candidate list:", bx.filter(pl.col("g").is_not_null())["q"].n_unique())
bx.write_parquet(f"{T}/exact_elsewhere.parquet")
smp = bad.sample(30, seed=3).join(ex, on="q").join(recs.select("q","qn","qa"), on="q").join(s1.select("s","sn","sa"), on="s").join(s1.select(sx="s", xn="sn", xa="sa"), on="sx")
print(smp.select("qn","sn","xn","qa","sa","xa"))
