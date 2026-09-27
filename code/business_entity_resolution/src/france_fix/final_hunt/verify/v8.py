import sys, re
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../same-address"))
from salib import *
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
V = rf"{SCRATCH}/final/verify/"
r = pl.read_parquet(V + "sa/acr_recs.parquet")
print(r.with_columns(nbb=pl.col("nS").clip(1, 6)).group_by("nbb").agg(recs=pl.len(), uniq_exact=pl.col("ex").sum()).sort("nbb"))
a = pl.read_parquet(rf"{SCRATCH}/final/same-address/add_france.parquet")
tw = pl.read_parquet(V + "sa/twins.parquet")
oo = pl.read_parquet(V + "sa/acr_oneoff.parquet").select("q", "s", "n_oneoff")
a = a.join(tw, on=["s", "q"], how="left").join(oo, on=["s", "q"], how="left").with_columns(pl.col("n_oneoff").fill_null(0))
dom = pl.col("nops").str.contains("n_domain")
a = a.with_columns(dom_exact=pl.Series([squash(q, True) == squash(s) for q, s in zip(a["qn"], a["sn"])]))
safe = a.filter(
    (pl.col("acr") & (pl.col("nb") == 1)) |
    (~pl.col("acr") & (~dom | pl.col("dom_exact")) & (pl.col("street_twins") == 0) & ~pl.col("aops").str.contains("a_chg:place|a_drop:place|a_chg:street")))
safe = safe.filter(pl.col("s1_nmatch") > 0)   # never turn an empty S1 row non-empty
print("safe subset", safe.height, " acronym", safe["acr"].sum(), " non-acronym", (~safe["acr"]).sum())
print("dropped: acronym nb>1", a.filter(pl.col("acr") & (pl.col("nb") > 1)).height, "; partial domains", a.filter(~pl.col("acr") & dom & ~pl.col("dom_exact")).height,
      "; empty S1 rows", (a["s1_nmatch"] == 0).sum())
safe.write_parquet(V + "same-address_france_same_address_noise_name_safe.parquet")
print(pl.read_parquet(V + "same-address_france_same_address_noise_name_safe.parquet").schema)
