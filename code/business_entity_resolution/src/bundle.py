"""Pack the inputs of ce_folds.py for another machine (no raw data, only what the transformers read).

  python bundle.py            -> WORK/bundle_ce_x.zip  (unzip into the other machine's repo root)

Contents (paths relative to the repo root):
  work/{train,test}_s{1,2,3}.parquet : entity_id, business_name, business_address of the records and S1 rows that
                                       appear in a close call (only these are read by rerank.py)
  work/ce_x/{train,test}_rows.parquet, work/ce_x/select.json : the close calls of v6's model + stage-1 ranks 3-5
"""
import os
import shutil
import zipfile
import polars as pl
from common import WORK, log, id_to_int


def main():
    """Write WORK/bundle_ce_x.zip: the ce_x close-call tables, and the entity_id, name and address of only the
    records and S1 rows that appear in them."""
    src = os.path.join(WORK, "ce_x")
    stage = os.path.join(WORK, "bundle_stage", "work")
    shutil.rmtree(os.path.join(WORK, "bundle_stage"), ignore_errors=True)
    os.makedirs(os.path.join(stage, "ce_x"))
    for f in ("train_rows.parquet", "test_rows.parquet", "select.json"):
        shutil.copy(os.path.join(src, f), os.path.join(stage, "ce_x", f))
    for split in ("train", "test"):
        rows = pl.read_parquet(os.path.join(src, f"{split}_rows.parquet"), columns=["q", "s"])
        ids = pl.concat([rows.select(id=pl.col("q").cast(pl.Int64)), rows.select(id=pl.col("s").cast(pl.Int64))]).unique()
        for k in (1, 2, 3):
            d = (pl.scan_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"))
                 .select("entity_id", "business_name", "business_address")
                 .with_columns(id=id_to_int("entity_id")).join(ids.lazy(), on="id").drop("id").collect())
            d.write_parquet(os.path.join(stage, f"{split}_s{k}.parquet"))
            log(f"{split}_s{k}: {d.height} rows")
    out = os.path.join(WORK, "bundle_ce_x.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:     # parquet is already compressed
        for root, _, files in os.walk(os.path.join(WORK, "bundle_stage")):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.join(WORK, "bundle_stage")))
    shutil.rmtree(os.path.join(WORK, "bundle_stage"))
    log(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
