# US train reference: does "record's exact name exists as another S1 in same city" change the truth rate of swap|nsame best candidates?
import polars as pl, sys, re, unicodedata, os
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from france_cal import toks
from common import WORK, id_to_int, read_truth
T = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/frfix/twins"
F = r"C:/Users/kanav/.claude/jobs/ef6f152d/tmp/france"
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(320)
def nrm(x):
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", x).strip()
def city_us(a):
    parts = [nrm(p) for p in (a or "").split(",")]
    c = [p for p in parts if p and not re.search(r"\d", p) and not re.fullmatch(r"[a-z]{2}", p)]
    return c[-1] if c else ""
s1 = pl.scan_parquet(os.path.join(WORK, "train_s1.parquet")).filter(pl.col("country") == "US").select(s=id_to_int("entity_id"), sn="business_name", sa="business_address").collect()
s1 = s1.with_columns(ks=pl.Series([" ".join(sorted(set(toks(x)))) for x in s1["sn"].to_list()]), scity=pl.Series([city_us(a) for a in s1["sa"].to_list()]))
u = pl.read_parquet(f"{F}/usi_top.parquet").filter(pl.col("c") == "US")
u = u.with_columns(kq=pl.Series([" ".join(sorted(set(toks(x)))) for x in u["qn"].to_list()]), qcity=pl.Series([city_us(a) for a in u["qa"].to_list()]))
cc = s1.group_by("ks", "scity").agg(nx_city=pl.len(), sx=pl.col("s"))
u = u.join(cc, left_on=["kq", "qcity"], right_on=["ks", "scity"], how="left").with_columns(pl.col("nx_city").fill_null(0))
tr = read_truth().select(ts=id_to_int("s1_id"), q=id_to_int("q_id"))
u = u.join(tr, on="q", how="left")
u = u.with_columns(nk=pl.col("pat").str.split("|").list.first(), num=pl.col("pat").str.split("|").list.last(),
                   true_in_x=pl.col("sx").list.contains(pl.col("ts")).fill_null(False))
print("US city parse sample:", u.select("qa", "qcity").head(3).to_dicts())
print(u.group_by("nk", "num").agg(n=pl.len(), rate=pl.col("label").mean(), ex=(pl.col("nx_city") > 0).mean(),
     rate_ex=pl.col("label").filter(pl.col("nx_city") > 0).mean(), rate_noex=pl.col("label").filter(pl.col("nx_city") == 0).mean(),
     true_to_x=pl.col("true_in_x").filter(pl.col("nx_city") > 0).mean(), none_ex=pl.col("ts").is_null().filter(pl.col("nx_city") > 0).mean()).sort("n", descending=True).head(14))
u.drop("sx").write_parquet(f"{T}/us_top_k.parquet")
