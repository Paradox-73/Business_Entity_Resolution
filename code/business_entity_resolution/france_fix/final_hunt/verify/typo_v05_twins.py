"""Other France S1 rows with the same core name as the proposed S1 (anywhere): does one of them fit the record's address as well or better?"""
import os, sys, re
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from rapidfuzz import fuzz
from common import WORK, id_to_int
from ops import words, undot_legal, LEGAL, strip_acc

pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(320); pl.Config.set_tbl_cols(25)
x = pl.read_parquet(r"C:/ber_scratch/final/fr-gathik-rest/fr_typo_in.parquet")


def core(n):
    return " ".join(sorted(w for w in words(undot_legal(n or "")) if w not in LEGAL))


def an(a):
    return " ".join(sorted(re.split(r"[^a-z0-9]+", strip_acc(a or "").lower()))).strip()


def nums(a):
    return set(re.findall(r"(?<![0-9])0*(\d{1,4})(?![0-9])", re.sub(r"(?<![0-9])\d{5}(?![0-9])", " ", a or "")))


s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id").cast(pl.Int64), on_="business_name", oa="business_address").collect()
s1 = s1.with_columns(core=pl.Series([core(n) for n in s1["on_"].to_list()]))
x = x.join(s1.select("s", "core"), on="s", how="left")
tw = x.select("s", "q", "core", "qa", "sa", "qn", "sn").join(s1.rename({"s": "s_o"}), on="core").filter(pl.col("s_o") != pl.col("s"))
print("pairs whose S1 core name has other France S1 rows:", tw["q"].n_unique(), "of", x.height, "; twin rows", tw.height)
rows = []
for r in tw.iter_rows(named=True):
    qa, sa, oa = r["qa"] or "", r["sa"] or "", r["oa"] or ""
    rows.append(dict(sim_s=fuzz.token_set_ratio(an(qa), an(sa)), sim_o=fuzz.token_set_ratio(an(qa), an(oa)),
                     num_s=bool(nums(qa) & nums(sa)), num_o=bool(nums(qa) & nums(oa))))
tw = pl.concat([tw, pl.DataFrame(rows)], how="horizontal")
tw = tw.with_columns(o_better=(pl.col("sim_o") >= pl.col("sim_s")) | (pl.col("num_o") & ~pl.col("num_s")))
g = tw.group_by("s", "q").agg(n_tw=pl.len(), any_better=pl.col("o_better").any(), best_o=pl.col("sim_o").max(), sim_s=pl.col("sim_s").first())
print("pairs with a same-core S1 fitting the record address as well or better:", g.filter(pl.col("any_better")).height)
print(tw.filter(pl.col("o_better")).select("sn", "qn", "on_", "qa", "sa", "oa", "sim_s", "sim_o").head(30))
# address agreement record vs proposed S1 in general
rows = [dict(sim=fuzz.token_set_ratio(an(a), an(b)), num=bool(nums(a) & nums(b)), qnum=len(nums(a)) > 0) for a, b in zip(x["qa"].fill_null("").to_list(), x["sa"].fill_null("").to_list())]
x = pl.concat([x, pl.DataFrame(rows)], how="horizontal")
print(x.select(pl.col("sim").cut([50, 70, 85, 95]).alias("b")).group_by("b").len().sort("b"))
print("house number shared", x["num"].sum(), "record has a number", x["qnum"].sum())
print(x.filter(pl.col("sim") < 70).select("sn", "qn", "sa", "qa", "sim", "opsF").head(20))
x = x.join(g.select("s", "q", "n_tw", "any_better"), on=["s", "q"], how="left").with_columns(pl.col("n_tw").fill_null(0), pl.col("any_better").fill_null(False))
x.select("s", "q", "sim", "num", "n_tw", "any_better").write_parquet(r"C:/ber_scratch/final/verify/typo_twins.parquet")
