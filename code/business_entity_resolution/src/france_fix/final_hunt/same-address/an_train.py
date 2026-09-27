import os, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salib import *
from common import log
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
j = pl.read_parquet("train_cands.parquet")
CLEAN = set("n_domain n_acronym n_squash n_word_order n_leet n_upper n_lower n_case_other n_title n_acc_add n_acc_strip n_legal_drop n_legal_dot n_legal_abbrev n_dblspace n_bracket n_hyphen n_comma_add n_comma_drop n_amp_swap n_idtag n_word_dup n_tradingas".split())
j = j.with_columns(nl=pl.col("nops").str.extract_all(r"n_[a-z_]+"))
j = j.with_columns(clean=pl.col("nl").list.eval(pl.element().is_in(list(CLEAN))).list.all(), nm=pl.col("nl").list.join("+"))
j = j.with_columns(n_clean=(pl.col("ok") & pl.col("clean")).sum().over("q"))
k = j.filter(pl.col("ok") & pl.col("clean") & (pl.col("n_clean") == 1) & (pl.col("h") < 500))
log(f"clean unique eval pairs {k.height} prec {k['y'].mean():.4f}")
log(str(k.group_by("country").agg(n=pl.len(), prec=pl.col("y").mean())))
log(str(k.group_by("nm").agg(n=pl.len(), prec=pl.col("y").mean(), low=pl.col("low").mean()).sort("n", descending=True)))
log(str(k.with_columns(num=pl.col("aops").str.extract(r"(a_num[a-z_0-9]*)")).group_by("num").agg(n=pl.len(), prec=pl.col("y").mean()).sort("n", descending=True)))
log(str(k.filter(~pl.col("y")).select("qn", "sn", "qa", "sa", "nops").head(40)))
k.write_parquet("train_clean.parquet")
