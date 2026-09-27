import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import WORK, id_to_int, read_tsv
import importlib.util
spec = importlib.util.spec_from_file_location("nzs", rf"{SCRATCH}/final/verify/nz_lib.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
R = os.path.dirname(WORK)
t = (pl.scan_parquet(rf"{SCRATCH}/france/fr_top.parquet")
     .filter(pl.col("pat").str.contains(r"^(swap|added|dropped|other)\|(nsame|nmiss)$"))
     .select("q", "s", "p2", "qn", "qa", "sn", "sa", "pat").collect())
print("in-list France best-candidate pairs with a name change, same/missing number:", t.height)
t = m.annotate(t, country="France")
t = t.filter(pl.col("sub").is_in(["amp_only", "contentdrop_inplace", "contentdrop_suffix", "noise_to_noise", "not_noise_in"]))
def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
for v in ("v7ens", "v10b"):
    p = os.path.join(R, "submissions", v, "matching_results.tsv")
    if os.path.exists(p):
        vv = pairs(p).with_columns(pl.lit(True).alias("in_" + v))
        t = t.join(vv, on=["s", "q"], how="left").with_columns(pl.col("in_" + v).fill_null(False))
ok = ~pl.col("lowbad")
cols = [c for c in t.columns if c.startswith("in_")]
print(t.with_columns(num=pl.col("pat").str.split("|").list.get(1)).group_by("sub", "num").agg(
    n=pl.len(), low_rate=pl.col("lowr").filter(ok).mean(), n_ok=ok.sum(), up=pl.col("up").sum(), down=pl.col("down").sum(),
    **{c: pl.col(c).mean() for c in cols}, low_acc10=pl.col("lowr").filter(ok & pl.col(cols[-1])).mean(),
    low_rej10=pl.col("lowr").filter(ok & ~pl.col(cols[-1])).mean()).sort("sub", "num"))
t.write_parquet(rf"{SCRATCH}/final/verify/nz_inlist.parquet")
