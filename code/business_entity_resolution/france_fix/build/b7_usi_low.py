"""US/India labels: lowercase share of record names by pattern and label (is lowercase a decoy fingerprint beyond swap/added?)."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
u = pl.read_parquet(f"{FR}/usi_top.parquet", columns=["qn", "label", "p", "pat", "country"])
u = u.filter(pl.col("qn").str.contains(r"[A-Za-z]")).with_columns(low=pl.col("qn") == pl.col("qn").str.to_lowercase(),
    kind=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last())
print(u.group_by("country", "kind", "num", "label").agg(n=pl.len(), low=(pl.col("low").mean()*100).round(2), p=pl.col("p").mean().round(3))
      .filter(pl.col("n") >= 200).sort("country", "kind", "num", "label"))
