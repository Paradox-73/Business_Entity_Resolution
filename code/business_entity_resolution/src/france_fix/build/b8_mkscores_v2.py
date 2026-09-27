"""V2 scores: descveto scores + France additions (add_set_v2 filtered) raised to p2 = max(p2, 0.95)."""
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
BASE = f"{SCRATCH}/frfix/namechg/test_scores_frmin_descveto.parquet"
OUT = f"{SCRATCH}/frfix/build/test_scores_frmin_descveto_gnadd.parquet"
add = pl.read_parquet("add_set_v2.parquet")
add = add.filter((pl.col("p3") >= 0.01) & (pl.max_horizontal("p3", "p2g") >= 0.1))
print("add set after score filter:", add.height, add.group_by("cls", "in7m").len().sort("cls", "in7m").to_dicts())
add.write_parquet("add_set_v2_final.parquet")
b = pl.read_parquet(BASE)
n0 = b.height
b = b.join(add.select("q", "s").with_columns(addf=pl.lit(True)), on=["q", "s"], how="left")
print("matched add rows:", b["addf"].sum())
b = b.with_columns(p2=pl.when(pl.col("addf").fill_null(False)).then(pl.max_horizontal("p2", pl.lit(0.95, pl.Float32))).otherwise(pl.col("p2")).cast(pl.Float32)).drop("addf")
assert b.height == n0
b.write_parquet(OUT)
old = pl.read_parquet(BASE)
print("rows with p2 changed:", (old["p2"] != b["p2"]).sum())
