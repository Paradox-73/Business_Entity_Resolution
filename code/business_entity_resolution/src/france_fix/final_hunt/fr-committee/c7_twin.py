"""Twin structure: number of same-country S1 rows sharing name_core (n_same) for train analog and France candidates.
Train: where do false committee pairs' records really belong (another S1 or none)?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int, read_truth
T = rf"{SCRATCH}/final/fr-committee"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40)
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]


def forms(col):
    t = (pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
         .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    return t.str.split(" ").list.eval(pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
        pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique().list.sort().list.join(" ")


def s1tab(split):
    d = pl.read_parquet(os.path.join(WORK, f"{split}_s1.parquet"), columns=["entity_id", "country", "name_core", "business_name"]).select(
        s=id_to_int("entity_id").cast(pl.Int64), country="country", nc="name_core", sf=forms("business_name"))
    d = d.join(d.group_by("country", "nc").agg(n_same=pl.len()), on=["country", "nc"])
    return d.join(d.group_by("country", "nc", "sf").agg(n_samef=pl.len()), on=["country", "nc", "sf"])


# ---- train analog
a = pl.read_parquet(os.path.join(T, "train_committee.parquet")).filter(pl.col("kind") == "add")
st = s1tab("train")
truth = read_truth().select(ts=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))
a = a.join(st.select("s", "n_same", "n_samef"), on="s", how="left").join(truth, on="q", how="left")
a = a.with_columns(fate=pl.when(pl.col("y")).then(pl.lit("true")).when(pl.col("ts").is_null()).then(pl.lit("no_s1")).otherwise(pl.lit("other_s1")),
                   tw=pl.col("n_same") > 1, amiss=pl.col("ops").str.contains("a_missing"))
print("TRAIN analog add: fate by class"); print(a.group_by("cls", "fate").len().pivot(on="fate", index="cls", values="len"))
print(a.group_by("cls", "tw").agg(n=pl.len(), prec=pl.col("y").mean().round(3)).sort("cls", "tw"))
print(a.group_by("cls", "amiss", "tw").agg(n=pl.len(), prec=pl.col("y").mean().round(3)).sort("cls", "amiss", "tw"))
u = a.filter(~pl.col("tw") & pl.col("cls").is_in(["noword", "word_other"]))
print("unique-name noword/word_other by pg band"); print(u.with_columns(pgb=pl.col("pg").cut([0.8, 0.9, 0.95, 0.99])).group_by("cls", "pgb").agg(n=pl.len(), prec=pl.col("y").mean().round(3)).sort("cls", "pgb"))
print(u.with_columns(pob=pl.col("po").cut([0.2, 0.4, 0.5])).group_by("cls", "pob").agg(n=pl.len(), prec=pl.col("y").mean().round(3)).sort("cls", "pob"))
# ---- France
c = pl.read_parquet(os.path.join(T, "cand.parquet")).filter(pl.col("kind") == "add")
c = c.with_columns(cls=pl.when(pl.col("ops").str.contains("n_swap:desc|n_add:desc")).then(pl.lit("desc"))
                   .when(pl.col("ops").str.contains("n_swap:|n_add:|n_drop:")).then(pl.lit("word_other"))
                   .when(pl.col("ops").str.contains("n_legal_change|n_legal_add")).then(pl.lit("legal"))
                   .otherwise(pl.lit("noword")), amiss=pl.col("ops").str.contains("a_missing"),
                   up=pl.col("ops").str.contains("a_num_up"), down=pl.col("ops").str.contains("a_num_down"),
                   lowok=~pl.col("ops").str.contains("n_domain|n_squash"))
sf = s1tab("test").filter(pl.col("country") == "France")
c = c.join(sf.select("s", "n_same", "n_samef"), on="s", how="left").with_columns(tw=pl.col("n_same") > 1, twf=pl.col("n_samef") > 1)
print("FRANCE committee add")
print(c.group_by("cls", "amiss", "tw").agg(n=pl.len(), low=pl.col("low").filter(pl.col("lowok")).mean().round(4), up=pl.col("up").mean().round(3),
                                          dn=pl.col("down").mean().round(3), po=pl.col("po").mean().round(3), pg=pl.col("pg").mean().round(3)).sort("cls", "amiss", "tw"))
# France reference: v10b matched pairs twin share (from calib parquet)
cal = pl.read_parquet(os.path.join(T, "calib.parquet")).join(sf.select("s", "n_same"), on="s", how="left").with_columns(tw=pl.col("n_same") > 1, amiss=pl.col("ops").str.contains("a_missing"))
print(cal.group_by("src", "cls", "tw").agg(n=pl.len(), low=pl.col("low").filter(pl.col("lowok")).mean().round(4), up=pl.col("ops").str.contains("a_num_up").mean().round(3),
                                          amiss=pl.col("amiss").mean().round(3)).sort("cls", "src", "tw"))
c.write_parquet(os.path.join(T, "cand2.parquet"))
