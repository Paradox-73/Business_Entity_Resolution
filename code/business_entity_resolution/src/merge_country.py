"""Replace one country's rows of a submission with the rows of another submission.

  python merge_country.py <base.tsv> <donor.tsv> <country> <out.tsv>

Used to change one country at a time so a leaderboard difference measures exactly that change
(e.g. v4 = v4 model for US/India + v3 rows for France)."""
import os
import sys
import polars as pl
from common import WORK, read_tsv

base, donor, country, out = sys.argv[1:5]
ids = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"])
keep = set(ids.filter(pl.col("country") == country)["entity_id"].to_list())
b, d = read_tsv(base), read_tsv(donor)
col0 = b.columns[0]
d = d.filter(pl.col(col0).is_in(list(keep)))
m = pl.concat([b.filter(~pl.col(col0).is_in(list(keep))), d]).with_columns(pl.col(b.columns[1]).fill_null(""))
assert m.height == b.height == ids.height, (m.height, b.height, ids.height)
assert d.height == len(keep), (d.height, len(keep))
m.write_csv(out, separator="\t", quote_style="never")
print(f"{out}: {m.height} rows; {country} rows ({d.height}) from {donor}; non-empty {(m[b.columns[1]] != '').sum()}")
