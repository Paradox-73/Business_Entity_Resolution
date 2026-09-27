"""Does Gathik's model (held-out scores, US/India labels) filter out all-lowercase decoys among word-change pairs?
If accepted false pairs are almost never lowercase, the lowercase test on a gathik-accepted pair set is circular."""
import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, read_truth, id_to_int
import importlib.util
spec = importlib.util.spec_from_file_location("nzs", r"C:/ber_scratch/final/verify/nz_lib.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
G = os.path.join(WORK, "gathik", "v9")
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
oof = i64(pl.read_parquet(os.path.join(G, "models_full_xgb_cons_oof.parquet"), columns=["q", "s", "p2"]))
s3 = i64(pl.read_parquet(os.path.join(G, "ce_x_oof_s3_bgef0bgef1bgef2.parquet"), columns=["q", "s", "p3"]))
touched = s3.select("q").unique().with_columns(t=pl.lit(True))
d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
d = d.select("q", "s", pg=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0)).otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))
del oof, s3
truth = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)).with_columns(y=pl.lit(True))
d = d.join(truth, on=["q", "s"], how="left").with_columns(pl.col("y").fill_null(False))
print("pairs", d.height)
fa = d.filter(~pl.col("y") & (pl.col("pg") >= 0.8)).with_columns(grp=pl.lit("false_acc_pg>=0.8"))
fr = d.filter(~pl.col("y") & (pl.col("pg") >= 0.05) & (pl.col("pg") < 0.5)).sample(80000, seed=1).with_columns(grp=pl.lit("false_rej_0.05-0.5"))
ta = d.filter(pl.col("y") & (pl.col("pg") >= 0.8)).sample(80000, seed=1).with_columns(grp=pl.lit("true_acc_pg>=0.8"))
tr_ = d.filter(pl.col("y") & (pl.col("pg") < 0.5)).with_columns(grp=pl.lit("true_rej_pg<0.5"))
tr_ = tr_.sample(min(40000, tr_.height), seed=1)
x = pl.concat([fa, fr, ta, tr_]); del d
print(x.group_by("grp").len())
s1 = pl.scan_parquet(os.path.join(WORK, "train_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), sn="business_name", sa="business_address", country="country").join(x.select("s").unique().lazy(), on="s").collect()
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet")).select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address")
                 .join(x.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
x = x.join(s1, on="s").join(rec, on="q")
x = m.annotate(x)
x.write_parquet(r"C:/ber_scratch/final/verify/nz_train_sel.parquet")
ok = ~pl.col("lowbad") & pl.col("wordchg")
print("word-change pairs (no domain/squash/single token): lowercase share by group")
print(x.filter(ok).group_by("grp").agg(n=pl.len(), low=pl.col("lowr").sum(), low_rate=pl.col("lowr").mean()).sort("grp"))
print("by group x subclass")
print(x.filter(ok).group_by("grp", "sub").agg(n=pl.len(), low=pl.col("lowr").sum(), low_rate=pl.col("lowr").mean()).sort("sub", "grp"))
