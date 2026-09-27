"""Best-candidate table of labelled US/India train records -> $BER_SCRATCH/france/usi_top.parquet.

S1 rows with hash(s) % 10 == 0 (10% of train S1 rows) and their records; p = out-of-fold stage-3 p3
(WORK/ce_x/oof_s3_smallfolds.parquet) where present, else the out-of-fold GBDT p2 (WORK/models/full_cons/oof.parquet).
One row per record: its best candidate, the label and the texts (france_cal.attach_text).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
import france_cal as fc
OUT = f"{SCRATCH}/france"
t0 = time.time()
s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country"]).select(s=id_to_int("entity_id"), c="country")
ss = s1.filter((pl.col("s").hash(seed=11) % 10) == 0)
g = pl.scan_parquet(os.path.join(WORK, "models", "full_cons", "oof.parquet")).select("q", "s", "p1", "p2", "label").join(ss.lazy(), on="s").collect().rename({"p2": "p2g"})
x = pl.scan_parquet(os.path.join(WORK, "ce_x", "oof_s3_smallfolds.parquet")).select("q", "s", "p3").join(ss.lazy().select("s"), on="s").collect()
b = g.join(x, on=["q", "s"], how="left").with_columns(p=pl.coalesce("p3", "p2g"))
print("pairs", b.height, "p3 present", b["p3"].is_not_null().sum(), time.time() - t0)
top = b.sort("p", descending=True).unique("q", keep="first")
top = fc.attach_text(top, "train")
top.write_parquet(f"{OUT}/usi_top.parquet")
print("wrote", top.height, time.time() - t0)
