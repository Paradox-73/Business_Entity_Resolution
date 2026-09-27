"""Validate the same-address generator on labelled train: records our held-out prediction leaves unmatched (all S1 searched,
pairs scored on the eval half h<500). Precision per op-group + macro-F0.5 gain on the eval half."""
import os, sys, zlib, time
sys.path.insert(0, r"C:/ber_scratch/final/same-address")
from salib import *
from common import WORK, read_truth, id_to_int, log, macro_f05_df
from pipeline import decide_expf
OUT = r"C:/ber_scratch/final/same-address"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
t0 = time.time()


def held(oof_path, s3_path):
    oof = i64(pl.read_parquet(oof_path, columns=["q", "s", "p2"]))
    s3 = i64(pl.read_parquet(s3_path, columns=["q", "s", "p3"]))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", p2=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0)).otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))


s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address",
    h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
evs = s1.filter(pl.col("h") < 500).select("s")
truth_all = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))
truth = truth_all.join(evs, on="s")
po_all = decide_expf(held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_bgefolds.parquet")), "p2", 0.5, 1.0).select("q", "s")
po = po_all.join(evs, on="s")
log(f"held-out prediction pairs {po_all.height}; eval-half {po.height}  ({time.time()-t0:.0f}s)")
rec = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet")).select("entity_id", "business_name", "business_address", "country") for k in (2, 3)])
         .select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address", country="country")
         .join(po_all.lazy().select("q").unique(), on="q", how="anti").collect())
log(f"unmatched train records {rec.height}  ({time.time()-t0:.0f}s)")
rec = add_key(rec, "qa")
s1k = add_key(s1, "sa")
log(f"keys built; rec with key {rec['k'].is_not_null().mean():.3f}, s1 with key {s1k['k'].is_not_null().mean():.3f}  ({time.time()-t0:.0f}s)")
# key recall on truth pairs (all truth, sample)
tp = truth_all.sample(200000, seed=1)
recall_src = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet")).select("entity_id", "business_address", "country") for k in (2, 3)])
                .select(q=id_to_int("entity_id").cast(pl.Int64), qa="business_address", country="country")
                .join(tp.lazy().select("q"), on="q").collect())
recall_src = add_key(recall_src, "qa", "kq")
kr = tp.join(recall_src, on="q").join(s1k.select("s", "k"), on="s")
log(f"key equality on true pairs: {(kr['kq'] == kr['k']).mean():.3f}")
del recall_src, kr
j = generate(rec.select("q", "qn", "qa", "country", "k"), s1k.select("s", "sn", "sa", "country", "k", "h"))
log(f"candidate pairs after prefilter {j.height}; ok {j['ok'].sum()}; keep {j['keep'].sum()}  ({time.time()-t0:.0f}s)")
j = j.join(truth_all.with_columns(y=pl.lit(True)), on=["q", "s"], how="left").with_columns(pl.col("y").fill_null(False))
j = j.with_columns(t_other=pl.col("q").is_in(truth_all["q"].implode()) & ~pl.col("y"))
j.write_parquet(os.path.join(OUT, "train_cands.parquet"))
k = j.filter(pl.col("keep") & (pl.col("h") < 500))
log(f"EVAL keep pairs {k.height}; precision {k['y'].mean():.4f}; record truly matched to another S1: {k['t_other'].mean():.4f}")
pl.Config.set_tbl_rows(50); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
log(str(k.group_by("country").agg(n=pl.len(), prec=pl.col("y").mean())))
k = k.with_columns(nm=pl.col("nops").str.extract_all(r"n_[a-z_]+").list.join("+"))
log(str(k.group_by("nm").agg(n=pl.len(), prec=pl.col("y").mean(), low=pl.col("low").mean()).sort("n", descending=True).head(40)))
log(str(k.with_columns(num=pl.col("aops").str.extract(r"(a_num[a-z_0-9]*)")).group_by("num").agg(n=pl.len(), prec=pl.col("y").mean()).sort("n", descending=True)))
N = evs.height
base = macro_f05_df(po, truth, evs)
for name, sub in [("keep all", k), ("keep US", k.filter(pl.col("country") == "US")), ("keep India", k.filter(pl.col("country") == "India"))]:
    if sub.height == 0:
        continue
    f = macro_f05_df(pl.concat([po.select("s", "q"), sub.select("s", "q")]), truth, evs)
    log(f"{name:14s} n {sub.height:6d} prec {sub['y'].mean():.4f}  dF {f - base:+.6f}  per pair x N {(f - base) * N / sub.height:+.4f}")
log(f"done {time.time()-t0:.0f}s")
