"""Average two 3-fold transformer families pair by pair into a new family (ce = mean of the two logits).

  python avg_family.py <fam_a> <fam_b> <new_fam>      e.g. python avg_family.py small base sb
Reads and writes WORK/ce_x/{train,test}_ce_<fam>f{0,1,2}.parquet; then: bash runners/build_combo.sh <name> <new_fam>.
v7g2 (26 Sep): sb = small + base, held-out halves protocol 0.98844 (v7g, stage 3 per family then blend: 0.98845).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import polars as pl
from common import WORK

D = os.environ.get("BER_CE_DIR", os.path.join(WORK, "ce_x"))


def main(fa, fb, out):
    for k in (0, 1, 2):
        for sp in ("train", "test"):
            a = pl.read_parquet(os.path.join(D, f"{sp}_ce_{fa}f{k}.parquet"))
            b = pl.read_parquet(os.path.join(D, f"{sp}_ce_{fb}f{k}.parquet"), columns=["q", "s", "ce"]).rename({"ce": "ce_b"})
            n = a.height
            a = a.join(b, on=["q", "s"], how="left")
            assert a.height == n and a["ce_b"].null_count() == 0, f"{sp} fold {k}: pairs of {fa} and {fb} differ"
            a = a.with_columns(ce=((pl.col("ce") + pl.col("ce_b")) / 2).cast(pl.Float32)).drop("ce_b")
            a.write_parquet(os.path.join(D, f"{sp}_ce_{out}f{k}.parquet"))
            print(f"{sp} fold {k}: {a.height} pairs -> {sp}_ce_{out}f{k}.parquet", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
