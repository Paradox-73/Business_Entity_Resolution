"""Independent label-free test: surface-noise features of the record (address/name formatting) by group.
Groups: TRUE refs (same-name nsame accepted; noise-word N nsame), DECOY refs (X class any number, G added ndiff),
D class nsame accepted (veto), D nsame rejected, D ndiff."""
import sys, re
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
NC = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"
OUT = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/refute_desc"
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "qn", "qa", "sn", "sa"])
a = pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "p3", "p2g", "acc", "in7m", "cls", "pat", "num"])
v = pl.read_parquet(f"{NC}/veto_set.parquet", columns=["q", "s"]).with_columns(veto=pl.lit(True))
a = a.join(t, on=["q", "s"]).join(v, on=["q", "s"], how="left").with_columns(pl.col("veto").fill_null(False),
        kind=pl.col("pat").str.split("|").list.first())
a = a.with_columns(
    grp=pl.when(pl.col("veto")).then(pl.lit("D_veto(acc)"))
    .when((pl.col("cls") == "D") & (pl.col("num") == "nsame") & pl.col("in7m")).then(pl.lit("D_nsame_restored"))
    .when((pl.col("cls") == "D") & (pl.col("num") == "nsame")).then(pl.lit("D_nsame_rej"))
    .when((pl.col("cls") == "D") & (pl.col("num") == "ndiff")).then(pl.lit("D_ndiff"))
    .when((pl.col("cls") == "samename") & (pl.col("num") == "nsame") & pl.col("acc")).then(pl.lit("T_same_nsame_acc"))
    .when((pl.col("cls") == "samename") & (pl.col("num") == "ndiff")).then(pl.lit("same_ndiff"))
    .when((pl.col("cls") == "N") & (pl.col("num") == "nsame")).then(pl.lit("T_N_nsame"))
    .when((pl.col("cls") == "X") & (pl.col("num") == "nsame")).then(pl.lit("X_nsame"))
    .when((pl.col("cls") == "X") & (pl.col("num") == "ndiff")).then(pl.lit("X_ndiff"))
    .when((pl.col("cls") == "G") & (pl.col("num") == "nsame")).then(pl.lit("G_nsame"))
    .when((pl.col("cls") == "G") & (pl.col("num") == "ndiff")).then(pl.lit("G_ndiff"))
    .otherwise(pl.lit("other")))
feat = dict(
    a_exact=pl.col("qa") == pl.col("sa"),
    a_ci=(pl.col("qa").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
          == pl.col("sa").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()),
    a_upper=pl.col("qa") == pl.col("qa").str.to_uppercase(),
    a_lower=pl.col("qa") == pl.col("qa").str.to_lowercase(),
    a_dblsp=pl.col("qa").str.contains("  "),
    a_len_ratio=pl.col("qa").str.len_chars() / pl.col("sa").str.len_chars(),
    a_postcode=pl.col("qa").str.contains(r"\b\d{5}\b"),
    s_postcode=pl.col("sa").str.contains(r"\b\d{5}\b"),
    n_upper=pl.col("qn") == pl.col("qn").str.to_uppercase(),
    n_lower=pl.col("qn") == pl.col("qn").str.to_lowercase(),
    n_dblsp=pl.col("qn").str.contains("  "),
    n_trail=pl.col("qn").str.contains(r"^\s|\s$"),
    n_bracket=pl.col("qn").str.contains(r"[\[\]]"),
    n_hyphen=pl.col("qn").str.contains("-") & ~pl.col("sn").str.contains("-"),
    n_formfront=pl.col("qn").str.contains(r"^(SARL|SAS|SASU|SA|EURL|SCI|SNC|sarl|sas|sasu|sa|eurl|sci|snc|Sarl|Sas|Eurl) ")
                & ~pl.col("sn").str.contains(r"^(SARL|SAS|SASU|SA|EURL|SCI|SNC) "),
    n_nform=pl.col("qn").str.to_lowercase().str.contains(r"\b(sarl|sas|sasu|sa|eurl|sci|snc|ei|eirl|selarl|scp|gie|earl)\b"),
    s_nform=pl.col("sn").str.to_lowercase().str.contains(r"\b(sarl|sas|sasu|sa|eurl|sci|snc|ei|eirl|selarl|scp|gie|earl)\b"),
)
a = a.with_columns(**feat)
a.select("q", "s", "grp", *feat.keys()).write_parquet(f"{OUT}/fr_feat.parquet")
r = a.group_by("grp").agg(n=pl.len(), **{k: pl.col(k).mean() for k in feat}).sort("grp")
print(r)
