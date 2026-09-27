"""Methodology section 5 ("Common false positives"): the raw texts of the France example pairs, the S1 rows at the
same address, and the first pairs of the street veto.

  python france_examples.py        (from any folder; prints; BER_SETS overrides the sets folder)

Reads WORK/test_s{1,2,3}.parquet and sets/fr_street_veto.parquet.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
import polars as pl  # noqa: E402
from common import WORK, id_to_int  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_width_chars(250)
SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sets"))
cols = ["entity_id", "business_name", "business_address"]
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=cols + ["country"]).with_columns(i=id_to_int("entity_id"))


def recs(ids):
    """Records (test S2/S3 rows) with integer ids in `ids`."""
    out = []
    for k in (2, 3):
        out.append(pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(cols)
                   .with_columns(i=id_to_int("entity_id")).filter(pl.col("i").is_in(ids)).collect())
    return pl.concat(out)


pairs = [(30551485150, 10822548516), (30596051876, 10664217230), (30598920048, 10815458842)]
q = recs([p[0] for p in pairs])
for qq, ss in pairs:
    r = q.filter(pl.col("i") == qq)
    s = s1.filter(pl.col("i") == ss)
    print("REC", r.select(cols).rows(), "\nS1 ", s.select(cols).rows())
    a = s["business_address"][0]
    print("  S1 rows at the same address:", s1.filter(pl.col("business_address") == a).select(cols).rows())
v = pl.read_parquet(os.path.join(SETS, "fr_street_veto.parquet")).head(3)
q = recs(v["q"].to_list())
for qq, ss in zip(v["q"], v["s"]):
    print("VETO REC", q.filter(pl.col("i") == qq).select(cols).rows(),
          "\n     S1 ", s1.filter(pl.col("i") == ss).select(cols).rows())
