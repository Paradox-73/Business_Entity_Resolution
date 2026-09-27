import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
b = pl.read_parquet("C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/lbcal/pairs.parquet")
print(b.schema)
print("rank dist", b["rk"].value_counts().sort("rk"))
b = b.with_columns(nk=pl.col("pat").str.split("|").list.get(0), num=pl.col("pat").str.split("|").list.get(1))
for k in ["j", "gn", "v3"]:
    for lab, f in [("added", pl.col("in_" + k) & ~pl.col("in_ens")), ("removed", ~pl.col("in_" + k) & pl.col("in_ens"))]:
        d = b.filter(f)
        if d.height == 0: print(k, lab, 0); continue
        print(f"== v7{k} {lab}: {d.height}  rk1 {(d['rk']==1).sum()} mean g {d['g'].mean():.3f} p3 {d['p3'].mean():.3f} pens {d['pens'].mean():.3f} veto {d['veto'].sum()} samestreet {d['samestreet'].mean():.3f}")
        if d.height > 500:
            print(d.group_by("pat").agg(n=pl.len(), g=pl.col("g").mean(), p3=pl.col("p3").mean()).sort("n", descending=True).head(8))
# acc pairs in v7ens by pattern
d = b.filter(pl.col("in_ens"))
print("v7ens accepted by pattern")
print(d.group_by("pat").agg(n=pl.len(), g=pl.col("g").mean(), p3=pl.col("p3").mean(), pe=pl.col("pe").mean(), st=pl.col("samestreet").mean()).sort("n", descending=True).head(25))
r1 = b.filter(pl.col("rk") == 1)
print("rank1 by pattern: n, acc rate")
print(r1.group_by("pat").agg(n=pl.len(), acc=pl.col("in_ens").mean(), g=pl.col("g").mean(), p3=pl.col("p3").mean(), gge5=(pl.col("g")>=0.5).mean(), p3ge5=(pl.col("p3")>=0.5).mean()).sort("n", descending=True).head(25))
print("p3 == g share:", (b["p3"] == b["g"]).mean(), " rank1:", (r1["p3"] == r1["g"]).mean())
