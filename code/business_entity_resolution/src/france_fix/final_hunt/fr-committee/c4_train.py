"""US/India labelled analog of the committee rule (eval half of train): Gathik-accepted pairs inside our lists that our
held-out decision does not contain. Precision by ops class, number direction, prob bands; lowercase share by label."""
import os, sys, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from common import WORK, read_truth, id_to_int, log
from pipeline import decide_expf
from ops import ops
G = os.path.join(WORK, "gathik", "v9")
T = rf"{SCRATCH}/final/fr-committee"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def held(oof_path, s3_path):
    oof = i64(pl.read_parquet(oof_path, columns=["q", "s", "p2"]))
    s3 = i64(pl.read_parquet(s3_path, columns=["q", "s", "p3"]))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", p=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0)).otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32))


s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"])
s1 = s1.select(s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address",
               h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
ev = s1.filter(pl.col("h") < 500)
truth = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))
sm = held(os.path.join(WORK, "models", "full_cons", "oof.parquet"), os.path.join(WORK, "ce_b2", "oof_s3_bgefolds.parquet"))
ours_list = sm.select("q", "s")
po = decide_expf(sm.rename({"p": "p2"}), "p2", 0.5, 1.0)
gat = held(os.path.join(G, "models_full_xgb_cons_oof.parquet"), os.path.join(G, "ce_x_oof_s3_bgef0bgef1bgef2.parquet"))
pg = decide_expf(gat.rename({"p": "p2"}), "p2", 0.5, 1.0).join(gat.rename({"p": "pg"}), on=["q", "s"])
del gat
com = (pg.join(ev.select("s", "country"), on="s").join(ours_list, on=["q", "s"], how="semi").join(po, on=["q", "s"], how="anti")
         .join(po.select("q", vs="s"), on="q", how="left").join(sm.rename({"p": "po"}), on=["q", "s"], how="left")
         .with_columns(kind=pl.when(pl.col("vs").is_null()).then(pl.lit("add")).otherwise(pl.lit("move")))
         .join(truth.with_columns(y=pl.lit(True)), on=["q", "s"], how="left").with_columns(pl.col("y").fill_null(False)))
nv = po.group_by("s").agg(n_v=pl.len())
com = com.join(nv, on="s", how="left").with_columns(pl.col("n_v").fill_null(0))
log(f"committee analog {com.height}; prec {com['y'].mean():.4f}")
log(str(com.group_by("kind").agg(n=pl.len(), prec=pl.col("y").mean())))
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"train_s{k}.parquet")).select("entity_id", "business_name", "business_address") for k in (2, 3)]).select(
    q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address").join(com.select("q").unique().lazy(), on="q").collect()
a = com.join(s1.select("s", "sn", "sa"), on="s").join(rec, on="q")
a = a.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or ""))) for r in a.iter_rows(named=True)], dtype=pl.Utf8))
a = a.with_columns(nm=pl.col("ops").str.extract_all(r"n_[a-z_]+(?::[a-z]+)?").list.join("+"), num=pl.col("ops").str.extract(r"(a_num[a-z0-9_]*)"),
                   low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"),
                   lowok=~pl.col("ops").str.contains("n_domain|n_squash"),
                   cls=pl.when(pl.col("ops").str.contains("n_swap:desc|n_add:desc")).then(pl.lit("desc"))
                   .when(pl.col("ops").str.contains("n_swap:|n_add:|n_drop:")).then(pl.lit("word_other"))
                   .when(pl.col("ops").str.contains("n_legal_change|n_legal_add")).then(pl.lit("legal"))
                   .otherwise(pl.lit("noword")),
                   up=pl.col("ops").str.contains("a_num_up"), down=pl.col("ops").str.contains("a_num_down"))
a.write_parquet(os.path.join(T, "train_committee.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
ag = lambda d, k: d.group_by(k).agg(n=pl.len(), prec=pl.col("y").mean(), low=pl.col("low").filter(pl.col("lowok")).mean(),
                                    low_T=pl.col("low").filter(pl.col("lowok") & pl.col("y")).mean(), low_F=pl.col("low").filter(pl.col("lowok") & ~pl.col("y")).mean(),
                                    up=pl.col("up").mean(), down=pl.col("down").mean(), po=pl.col("po").mean(), pg=pl.col("pg").mean()).sort(k)
print(ag(a, ["kind", "cls"]))
print(ag(a.filter(pl.col("kind") == "add"), ["country", "cls"]))
aa = a.filter(pl.col("kind") == "add").with_columns(pob=pl.col("po").cut([0.1, 0.3, 0.5, 0.7]), pgb=pl.col("pg").cut([0.7, 0.8, 0.9, 0.95, 0.99]))
print(ag(aa, ["cls", "pob"]))
print(ag(aa, ["cls", "pgb"]))
print(ag(aa.filter(pl.col("cls") == "noword"), "num"))
print(ag(aa.filter(pl.col("cls") == "noword"), "nm").sort("n", descending=True).head(25))
print(ag(aa.filter(pl.col("cls") == "word_other"), "nm").sort("n", descending=True).head(25))
