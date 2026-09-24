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


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
