"""Early check of the 3-fold design: the new fold-0 transformer vs transformer A (v7ens) on the SAME held-out pairs
(eval-group close calls whose record is in fold 0; neither model trained on them)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss
from common import WORK as W  # noqa: E402
name = sys.argv[1] if len(sys.argv) > 1 else "small"
a = pl.read_parquet(f"{W}/ce/train_ce.parquet", columns=["q", "s", "label", "ce"]).rename({"ce": "ce_a"})
f = pl.read_parquet(f"{W}/ce_x/train_ce_{name}f0.parquet", columns=["q", "s", "ce"]).rename({"ce": "ce_f"})
d = a.join(f, on=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64))
y = d["label"].cast(pl.Int8).to_numpy()
print(f"common held-out pairs: {d.height}, positives {y.sum()}")
for c in ("ce_a", "ce_f"):
    x = d[c].to_numpy().astype(np.float64)
    p = 1 / (1 + np.exp(-x))
    top = d.with_columns(r=pl.col(c).rank("ordinal", descending=True).over("q")).filter(pl.col("r") == 1)
    has = d.group_by("q").agg(pl.col("label").max().alias("h")).filter("h")
    acc = top.join(has, on="q")["label"].mean()
    print(f"{c}: AUC {roc_auc_score(y, x):.5f}  log loss {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.5f}  "
          f"best candidate is the true S1 (records with one) {acc:.5f}")
