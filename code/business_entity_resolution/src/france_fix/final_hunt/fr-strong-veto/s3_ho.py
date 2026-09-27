import sys, zlib, polars as pl
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
W = f"{WORK}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
s1 = pl.read_parquet(W+"pairs/full/s1.parquet")
s1 = s1.with_columns(h=pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
ev = s1.filter(pl.col("h") < 500).select("s", "country")
g = pl.read_parquet(W+"models/full_cons/oof.parquet", columns=["q","s","p2","label"]).join(ev, on="s")
def s3(f, n):
    return pl.read_parquet(f, columns=["q","s","p3"]).join(ev.select("s"), on="s").rename({"p3": n})
d = g
for f, n in [(W+"ce_x/oof_s3_ab.parquet","ab"),(W+"ce_b2/oof_s3_smallfolds.parquet","sm"),(W+"ce_b2/oof_s3_bgefolds.parquet","ob"),
             (W+"ce_b2/oof_s3_e5lfolds.parquet","oe"),(W+"gathik/v9/ce_x_oof_s3_bgef0bgef1bgef2.parquet","gb")]:
    d = d.join(s3(f, n), on=["q","s"], how="left")
d = d.with_columns(pl.col("label").fill_null(False))
print("eval pairs", d.height, "true", d["label"].sum())
for weak in ["ab","sm"]:
    x = d.with_columns(pf=pl.min_horizontal("p2", pl.col(weak).fill_null(pl.col("p2"))))
    x = x.with_columns(best=pl.col("pf") == pl.col("pf").max().over("q"))
    acc = x.filter(pl.col("best") & (pl.col("pf") >= 0.5))
    print(f"--- weak={weak}: accepted {acc.height}, true share {acc['label'].mean():.4f}; weak-scored {acc.filter(pl.col(weak).is_not_null()).height}")
    both = acc.filter(pl.col("ob").is_not_null() & pl.col("gb").is_not_null())
    print("   both strong present", both.height, "true", round(both["label"].mean(),4))
    for t in [0.05,0.1,0.2,0.3,0.5]:
        a = both.filter((pl.col("ob")<t)&(pl.col("gb")<t))
        b = a.filter(pl.col("oe")<t)
        print(f"   t={t}: both-bge-low n={a.height} true={a['label'].mean() if a.height else float('nan'):.3f} | +e5l low n={b.height} true={b['label'].mean() if b.height else float('nan'):.3f}", 
              a.group_by("country").agg(pl.len(), pl.col("label").mean()).sort("country").rows())
    if weak == "ab":
        acc.filter(pl.col("ob").is_not_null() & pl.col("gb").is_not_null() & (pl.col("ob")<0.5)&(pl.col("gb")<0.5)).write_parquet(OUT+"ho_cand_ab.parquet")
    else:
        acc.filter(pl.col("ob").is_not_null() & pl.col("gb").is_not_null() & (pl.col("ob")<0.5)&(pl.col("gb")<0.5)).write_parquet(OUT+"ho_cand_sm.parquet")
