import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(330); pl.Config.set_tbl_cols(20)
T = rf"{SCRATCH}/final/fr-gathik-rest"
d = pl.read_parquet(os.path.join(T, "fr_noise_in.parquet"))
i64 = lambda x: x.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
qs = d.select("q").unique().lazy()
L = []
for f in ("test_scores_full_cons.parquet", "frfix3/test_scores_v9y_combo.parquet"):
    t = i64(pl.scan_parquet(os.path.join(WORK, f)).select("q", "s", "p2").join(qs, on="q").collect()).with_columns(src=pl.lit(f.split("/")[0][:12]))
    L.append(t)
ours = pl.concat(L)
comb = ours.filter(pl.col("src") != "test_scores_")
best = comb.sort("p2", descending=True).unique("q", keep="first").rename({"s": "s_o", "p2": "p_o"}).drop("src")
ncand = comb.group_by("q").agg(n_cand=pl.len())
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(pl.col("country") == "France").select(s_o=id_to_int("entity_id").cast(pl.Int64), on="business_name", oa="business_address")
x = d.join(best, on="q", how="left").join(ncand, on="q", how="left")
x = x.join(s1.join(x.select("s_o").unique().lazy(), on="s_o").collect(), on="s_o", how="left")
print("our best p on the record:", x["p_o"].describe())
print("records with no combo candidates:", x.filter(pl.col("s_o").is_null()).height)
print(x.select("sn", "qn", "on", "sa", "qa", "oa", "p2", "p_o").sample(40, seed=3))
x.write_parquet(rf"{SCRATCH}/final/verify/nz_ours.parquet")
