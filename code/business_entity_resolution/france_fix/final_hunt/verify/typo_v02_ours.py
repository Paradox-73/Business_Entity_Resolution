"""For the 532 records: which other S1s did OUR lists score them against, with what p, and what do those S1s look like."""
import os, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
from common import WORK, id_to_int

x = pl.read_parquet(r"C:/ber_scratch/final/fr-gathik-rest/fr_typo_in.parquet")
i64 = lambda t: t.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
qs = x.select("q").lazy()
parts = []
for name, p in [("cons", os.path.join(WORK, "test_scores_full_cons.parquet")), ("cex", os.path.join(WORK, "ce_x", "test_rows.parquet")),
                ("combo", os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet"))]:
    sc = pl.scan_parquet(p)
    cols = sc.collect_schema().names()
    t = i64(sc.join(qs.with_columns(pl.col("q").cast(sc.collect_schema()["q"])), on="q").collect())
    print(name, cols, t.height, t["q"].n_unique())
    keep = [c for c in ("p1", "p2") if c in cols]
    parts.append(t.select("q", "s", *[pl.col(c).alias(f"{c}_{name}") for c in keep]))
o = parts[0].join(parts[1], on=["q", "s"], how="full", coalesce=True).join(parts[2], on=["q", "s"], how="full", coalesce=True)
print("our candidate pairs for these records", o.height, "per record", o.group_by("q").len()["len"].describe())
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), on_="business_name", oa="business_address").join(
    o.select("s").unique().lazy(), on="s").collect()
o = o.join(s1, on="s", how="left")
best = o.sort("p2_combo", descending=True, nulls_last=True).unique("q", keep="first")
m = x.select("s", "q", "sn", "qn", "sa", "qa", "p2", "cls2").join(best.rename({"s": "s_our"}), on="q", how="left")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
print(m.select(pl.col("p2_combo").cut([0.05, 0.2, 0.5]).alias("b")).group_by("b").len().sort("b"))
print(m.select("sn", "qn", "on_", "p2_combo", "p2_cons", "sa", "oa").sort("p2_combo", descending=True, nulls_last=True).head(40))
print("our best other S1 same name as the proposed S1:", (m["on_"] == m["sn"]).sum())
m.write_parquet(r"C:/ber_scratch/final/verify/typo_ours.parquet")
