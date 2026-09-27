import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(60)
one = pl.read_parquet(f"{SCRATCH}/frfix3/refute_moreveto/one_swaps.parquet")
one = one.with_columns(low=pl.col("nc").list.contains("LOWER"),
    handle=~pl.col("qn").fill_null("").str.contains(" ") & pl.col("sn").fill_null("").str.contains(" "))
STEMS = {("sport","sportive"),("sportive","sport"),("college","collectif"),("collectif","college"),("groupe","groupement")}
one = one.with_columns(stemsw=pl.struct("d","a").map_elements(lambda r: (r["d"], r["a"]) in STEMS, return_dtype=pl.Boolean))
one = one.with_columns(grp=pl.when(pl.col("stemsw")).then(pl.lit("STEM"))
    .when((pl.col("cd")=="D") & (pl.col("ca")=="D")).then(pl.lit("D->D other"))
    .when((pl.col("cd").is_in(["D","G"])) & (pl.col("ca")=="D")).then(pl.lit("G->D"))
    .when((pl.col("ca")=="N")).then(pl.lit("x->N"))
    .when((pl.col("ca")=="rare") & (pl.col("cd")=="rare")).then(pl.lit("rare->rare"))
    .otherwise(pl.lit("other")))
def nb(n):
    return (pl.when(pl.col("num")=="NSAME").then(pl.lit("same")).when(pl.col("num").str.starts_with("UP")).then(pl.lit("up"))
            .when(pl.col("num").str.starts_with("DOWN")).then(pl.lit("down")).otherwise(pl.lit("miss")))
one = one.with_columns(nb=nb(0))
t = (one.filter(~pl.col("handle")).group_by("grp").agg(n=pl.len(), same=(pl.col("nb")=="same").mean(), up=(pl.col("nb")=="up").mean(),
      up1_20=pl.col("num").is_in(["UP1","UP2_5","UP6_20"]).mean(), down=(pl.col("nb")=="down").mean(), miss=(pl.col("nb")=="miss").mean(),
      low_all=pl.col("low").mean(), low_same=pl.col("low").filter(pl.col("nb")=="same").mean(), n_same=(pl.col("nb")=="same").sum(),
      nlow_same=pl.col("low").filter(pl.col("nb")=="same").sum(),
      acc=pl.col("acc").mean(), acc_same=pl.col("acc").filter(pl.col("nb")=="same").mean())
     .sort("n", descending=True))
print(t)
# per stem type
print(one.filter(pl.col("stemsw") & ~pl.col("handle")).group_by("d","a").agg(n=pl.len(), same=(pl.col("nb")=="same").mean(), up=(pl.col("nb")=="up").mean(),
     down=(pl.col("nb")=="down").mean(), low=pl.col("low").mean(), nlow=pl.col("low").sum(), acc=pl.col("acc").sum(), acc_same=pl.col("acc").filter(pl.col("nb")=="same").sum()))
