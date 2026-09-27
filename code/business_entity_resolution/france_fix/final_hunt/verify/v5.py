import os, sys, re
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from common import WORK, id_to_int, read_tsv
from ops import words, STOPW, LEGAL, undot_legal
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
D = r"C:/ber_scratch/final/same-address/"
k = pl.read_parquet(D + "fr_all_keep.parquet", columns=["q", "s", "qn", "sn", "nb", "nops", "v_s", "acr", "keep"]).filter(pl.col("keep") & pl.col("acr"))
g = (read_tsv(os.path.join(WORK, "gathik", "v8_matching_results.tsv")).with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
     .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))).with_columns(g_acc=pl.lit(True))
gp = i64(pl.read_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet"), columns=["q", "s", "p2"])).rename({"p2": "gp"})
frp = i64(pl.read_parquet(os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet"), columns=["q", "s", "p2"])).rename({"p2": "fr_p"})
k = k.join(g, on=["q", "s"], how="left").join(gp, on=["q", "s"], how="left").join(frp, on=["q", "s"], how="left").with_columns(pl.col("g_acc").fill_null(False))
core = lambda n: [w for w in words(undot_legal(n or "")) if w not in LEGAL and w not in STOPW]
k = k.with_columns(L=pl.col("sn").map_elements(lambda n: len(core(n)), return_dtype=pl.Int64),
                   grp=pl.when(pl.col("v_s").is_null()).then(pl.lit("unmatched")).when(pl.col("v_s") == pl.col("s")).then(pl.lit("matched_same")).otherwise(pl.lit("matched_other")))
print(k.group_by("grp").agg(n=pl.len(), L2=(pl.col("L") == 2).mean(), nb=pl.col("nb").mean(), nb1=(pl.col("nb") == 1).mean(), g_acc=pl.col("g_acc").mean(),
      gp_have=pl.col("gp").is_not_null().mean(), gp=pl.col("gp").mean(), frp_have=pl.col("fr_p").is_not_null().mean(), frp=pl.col("fr_p").mean()))
print(k.filter(pl.col("grp") != "matched_other").group_by("grp", "L").agg(n=pl.len(), g_acc=pl.col("g_acc").mean(), gp=pl.col("gp").mean(), frp=pl.col("fr_p").mean()).sort("grp", "L"))
print(k.filter(pl.col("grp") != "matched_other").with_columns(nbb=pl.col("nb").clip(1, 8)).group_by("grp", "nbb").agg(n=pl.len()).sort("nbb", "grp"))
