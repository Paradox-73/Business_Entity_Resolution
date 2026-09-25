"""Clean every source file once and save as parquet in WORK/{split}_s{k}.parquet."""
import json
import os
import sys
import polars as pl
from common import WORK, read_tsv, raw_path, log
from normalize import normalize


def main(splits=("train", "test")):
    maps = json.load(open(os.path.join(WORK, "maps.json"), encoding="utf8"))
    for split in splits:
        for k in (1, 2, 3):
            out = os.path.join(WORK, f"{split}_s{k}.parquet")
            if os.path.exists(out):
                continue
            df = read_tsv(raw_path(split, k))
            df = normalize(df, maps["script"], maps["abbrev"]).with_columns(src=pl.lit(k, pl.Int8))
            df.write_parquet(out)
            log(f"{split} S{k}: {df.height} rows -> {out}")
            del df


def france():
    """TEST France rows only, cleaned with the French rules -> WORK/testfr_s{k}.parquet (split name 'testfr')."""
    for k in (1, 2, 3):
        out = os.path.join(WORK, f"testfr_s{k}.parquet")
        df = read_tsv(raw_path("test", k)).filter(pl.col("country") == "France")
        df = normalize(df, french=True).with_columns(src=pl.lit(k, pl.Int8))
        df.write_parquet(out)
        log(f"testfr S{k}: {df.height} rows -> {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["france"]:
        france()
    else:
        main(sys.argv[1:] or ("train", "test"))
