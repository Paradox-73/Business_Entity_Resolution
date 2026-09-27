"""Methodology appendix B.1: held-out AUC of the three cross-encoder families (e5-small, bge-reranker-v2-m3,
e5-large) per fold, on all out-of-fold rows and on the close calls whose record does not span both S1 halves.

  python ce_auc.py        (from any folder; prints)

Reads WORK/ce/train_rows.parquet (close calls without ranks 3-5) and WORK/ce_b2/train_ce_{small,bge,e5l}f<k>.parquet
(out-of-fold logits of README steps 11, 12 and 21).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import WORK  # noqa: E402

FAMS = ("small", "bge", "e5l")
cc = pl.read_parquet(os.path.join(WORK, "ce", "train_rows.parquet"), columns=["q", "s"]).with_columns(cc=pl.lit(True))
for k in range(3):
    d = None
    for fam in FAMS:
        x = pl.read_parquet(os.path.join(WORK, "ce_b2", f"train_ce_{fam}f{k}.parquet"),
                            columns=["q", "s", "label", "grp", "ce"]).rename({"ce": fam})
        d = x if d is None else d.join(x.select("q", "s", fam), on=["q", "s"])
    if k == 0:
        print(d.group_by("grp").len())
    y = d["label"].to_numpy()
    print(k, d.height, {f: round(roc_auc_score(y, d[f].to_numpy()), 5) for f in FAMS})
    e = d.join(cc, on=["q", "s"]).filter(pl.col("grp") != "mixed")
    for g in e["grp"].unique().to_list():
        ee = e.filter(pl.col("grp") == g)
        yy = ee["label"].to_numpy()
        print("   close calls grp", g, ee.height, {f: round(roc_auc_score(yy, ee[f].to_numpy()), 5) for f in FAMS})
    yy = e["label"].to_numpy()
    print("   close calls no mixed", e.height, {f: round(roc_auc_score(yy, e[f].to_numpy()), 5) for f in FAMS})
