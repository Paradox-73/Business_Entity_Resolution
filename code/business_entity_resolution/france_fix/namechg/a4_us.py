"""US/India labelled: truth rate of same-number name changes by added word and by dropped word; keep rate per added word."""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from rapidfuzz import fuzz
import france_cal as fc
pl.Config.set_tbl_rows(150); pl.Config.set_tbl_width_chars(250)
FR = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
OUT = "C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/namechg"
u = pl.read_parquet(f"{FR}/usi_top.parquet", columns=["q", "s", "label", "p", "p3", "p2g", "qn", "sn", "pat", "country"])
u = u.with_columns(nk=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last())
u = u.filter(pl.col("nk").is_in(["swap", "added", "dropped", "other"]))


def diff(sn, qn):
    ts, tq = fc.toks(sn), fc.toks(qn)
    return ([w for w in tq if not any(fuzz.ratio(w, x) >= 80 for x in ts)],
            [w for w in ts if not any(fuzz.ratio(w, b) >= 80 for b in tq)])


r = [diff(a, b) for a, b in zip(u["sn"].to_list(), u["qn"].to_list())]
u = u.with_columns(added=pl.Series([x[0] for x in r], dtype=pl.List(pl.Utf8)), dropped=pl.Series([x[1] for x in r], dtype=pl.List(pl.Utf8)))
u.drop("qn", "sn").write_parquet(f"{OUT}/usi_chg.parquet")
x = u.filter(pl.col("nk").is_in(["swap", "added"]) & (pl.col("added").list.len() == 1)).with_columns(a=pl.col("added").list.first())
for c in ("US", "India"):
    y = x.filter(pl.col("country") == c)
    t = y.group_by("a").agg(n=pl.len(), keep=(pl.col("num") == "nsame").sum() / (pl.col("num") != "nmiss").sum(),
                            n_same=(pl.col("num") == "nsame").sum(),
                            true_same=pl.col("label").filter(pl.col("num") == "nsame").mean(),
                            true_diff=pl.col("label").filter(pl.col("num") == "ndiff").mean(),
                            true_swap_same=pl.col("label").filter((pl.col("num") == "nsame") & (pl.col("nk") == "swap")).mean(),
                            true_add_same=pl.col("label").filter((pl.col("num") == "nsame") & (pl.col("nk") == "added")).mean())
    print(c, "added word: keep rate, truth at same number")
    print(t.sort("n", descending=True).head(60))
    # truth by (swap at same number) for the swap kind: is dropped word a frequent S1 word
