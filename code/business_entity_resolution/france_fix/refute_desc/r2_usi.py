"""US/India labelled: do record casing / legal-form-position / double-space fingerprints separate true from false name-changed records?"""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
FR = "C:/ber_scratch/france"
u = pl.read_parquet(f"{FR}/usi_top.parquet", columns=["q", "s", "label", "p", "qn", "sn", "qa", "sa", "pat", "country"])
LF = r"(?i)^(inc|llc|ltd|corp|co|pvt|private|limited|llp|plc|the)\b"
u = u.with_columns(
    n_lower=pl.col("qn") == pl.col("qn").str.to_lowercase(),
    n_upper=pl.col("qn") == pl.col("qn").str.to_uppercase(),
    n_dblsp=pl.col("qn").str.contains("  "),
    n_formfront=pl.col("qn").str.contains(LF) & ~pl.col("sn").str.contains(LF),
    a_ci=(pl.col("qa").str.to_lowercase().str.replace_all(r"\s+", " ") == pl.col("sa").str.to_lowercase().str.replace_all(r"\s+", " ")),
    kind=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last())
r = (u.filter(pl.col("kind").is_in(["same", "swap", "added", "dropped", "other"]))
     .group_by("country", "kind", "num", "label").agg(n=pl.len(), n_lower=pl.col("n_lower").mean(), n_upper=pl.col("n_upper").mean(),
                                                      n_dblsp=pl.col("n_dblsp").mean(), n_formfront=pl.col("n_formfront").mean(), a_ci=pl.col("a_ci").mean())
     .filter(pl.col("n") >= 400).sort("country", "kind", "num", "label"))
print(r)
