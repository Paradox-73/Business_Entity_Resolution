"""Write the proposal sets: France adds (records unmatched in v10b), US/India adds (too few to propose), plus twin evidence."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salib import *
from common import WORK, id_to_int, log
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(220)
t = pl.read_parquet("test_cands.parquet").filter(pl.col("keep"))
# name twins: other S1 rows of the same country with the same squashed name (legal forms removed)
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name")
s1 = s1.with_columns(sq=pl.Series([squash(x) for x in s1["sn"].to_list()]))
tw = s1.group_by("country", "sq").len("n_twin")
t = t.with_columns(sq=pl.Series([squash(x) for x in t["sn"].to_list()])).join(tw, on=["country", "sq"], how="left").with_columns(
    acr=pl.col("nops").str.contains("n_acronym"))
fr = t.filter((pl.col("country") == "France") & ~pl.col("veto") & ~pl.col("g_other"))
ui = t.filter((pl.col("country") != "France") & ~pl.col("veto") & ~pl.col("g_other"))
log(f"France add {fr.height} (acronym {fr['acr'].sum()}); S1 empty in v10b {(fr['s1_nmatch'] == 0).sum()}; name twins>1 {(fr['n_twin'] > 1).sum()}")
log(str(fr.group_by("acr", tw=pl.col("n_twin") > 1).agg(n=pl.len(), g_acc=pl.col("g_acc").mean(), gp=pl.col("gp").mean(), in_list=pl.col("in_list").mean())))
log(f"US/India add {ui.height} {dict(ui.group_by('country').len().iter_rows())}")
cols = ["s", "q", "country", "qn", "qa", "sn", "sa", "nops", "aops", "acr", "nb", "n_twin", "s1_nmatch", "g_acc", "gp", "in_list", "fr_p", "low"]
fr.select(cols).write_parquet("add_france.parquet")
ui.select(cols).write_parquet("add_us_india_small.parquet")
# q must be unique and unmatched in v10b (checked upstream); sanity
assert fr["q"].n_unique() == fr.height and ui["q"].n_unique() == ui.height
