import os, sys, re, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../same-address"))
from salib import *
from common import WORK, id_to_int
from ops import words, STOPW, LEGAL, undot_legal
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(250)
D = rf"{SCRATCH}/final/same-address/"
t = pl.read_parquet(D + "test_cands.parquet").filter(pl.col("country") == "France")
# (1) visible legal-change look-alikes at same key (no word-level change)
lc = t.filter(pl.col("nops").str.contains("n_legal_change|n_legal_add") & ~pl.col("nops").str.contains("n_swap|n_add:|n_drop:|n_acronym"))
print("legal change/add only, same key, unmatched France records:", lc["q"].n_unique(), " pairs", lc.height)
sw1 = t.filter(pl.col("nops").str.contains("n_swap:") & ~pl.col("nops").str.contains("n_acronym"))
print("records with a word swap at same key:", sw1["q"].n_unique())
# (2) same-street twin: another France S1 with the same squashed name on the same street words but a different house number
a = pl.read_parquet(D + "add_france.parquet")
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
sq_t = set(squash(x) for x in a["sn"].to_list())
s1 = s1.with_columns(sq=pl.Series([squash(x) for x in s1["sn"].to_list()])).filter(pl.col("sq").is_in(list(sq_t)))
s1 = add_key(s1, "sa").with_columns(st=pl.col("k").str.split("|").list.get(1, null_on_oob=True))
a = a.with_columns(sq=pl.Series([squash(x) for x in a["sn"].to_list()]))
a = add_key(a, "sa").with_columns(st=pl.col("k").str.split("|").list.get(1, null_on_oob=True))
tw = a.select("s", "sq", "st").join(s1.select(s2="s", sq="sq", st="st"), on=["sq", "st"]).filter(pl.col("s2") != pl.col("s")).group_by("s").len("street_twins")
a = a.join(tw, on="s", how="left").with_columns(pl.col("street_twins").fill_null(0))
print("pairs whose S1 has a same-name twin on the same street (other number):", (a["street_twins"] > 0).sum())
print(a.group_by(pl.col("street_twins") > 0, "acr").agg(n=pl.len(), g_acc=pl.col("g_acc").mean(), gp=pl.col("gp").mean(), frp=pl.col("fr_p").mean()))
# (3) non-acronym part
na = a.filter(~pl.col("acr"))
print("non-acronym pairs", na.height, "g_acc", na["g_acc"].mean(), "in_list", na["in_list"].mean())
print(na.select("qn", "sn", "qa", "sa", "nops", "n_twin", "street_twins", "gp", "fr_p").sort("nops").head(60))
a.select("s", "q", "street_twins").write_parquet(rf"{SCRATCH}/final/verify/sa/twins.parquet")
