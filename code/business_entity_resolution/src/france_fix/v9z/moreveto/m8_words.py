"""Word-level scan: per added word (non-D, non-vetoed), lowercase share among the case-blind model's accepts; keep-rate bins."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(40)
OUT = f"{SCRATCH}/frfix3/moreveto"
NC = f"{SCRATCH}/frfix/namechg"
g = pl.read_parquet(f"{OUT}/grp_table2.parquet").filter(pl.col("pop") & ~pl.col("handle") & pl.col("added").is_not_null())
g = g.filter(pl.col("added").list.len() >= 1)
w = pl.read_parquet(f"{NC}/word_class.parquet").rename({"a": "word"})
e = g.select("q", "s", "cur", "lowx", "npat", "grp", word=pl.col("added")).explode("word").join(w, on="word", how="left")
e = e.with_columns(pl.col("cls").fill_null("rare"))
e = e.with_columns(kb=pl.when(pl.col("n") < 10).then(pl.lit("n<10")).otherwise(pl.col("keep").cut([0.08, 0.15, 0.25, 0.35, 0.5, 0.6, 0.7]).cast(pl.Utf8)),
                   db=pl.col("df_s1").fill_null(0).cut([10, 30, 100, 1000]).cast(pl.Utf8))
print("by class x keep bin x S1-df bin (added-word occurrences among accepted, excluding D-veto):")
r = e.group_by("cls", "kb", "db").agg(n=pl.len(), n_cur=pl.col("cur").sum(), nlow=pl.col("lowx").sum()).with_columns(low=pl.col("nlow") / pl.col("n")).sort("cls", "kb", "db")
print(r.filter(pl.col("n") >= 30))
wd = e.group_by("word", "cls").agg(n=pl.len(), n_cur=pl.col("cur").sum(), nlow=pl.col("lowx").sum(), keep=pl.col("keep").first(), nk=pl.col("n").first(), df=pl.col("df_s1").first())
print("words with >= 2 lowercase records:")
print(wd.filter(pl.col("nlow") >= 2).with_columns(low=pl.col("nlow") / pl.col("n")).sort("nlow", descending=True).head(50))
e.write_parquet(f"{OUT}/word_occ.parquet")
