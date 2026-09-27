"""Methodology sections 4 and 5: groups of the recovered France pairs, and the "Club" <-> "Ecole" swaps inside the
descriptor-word veto (230 pairs).

  python desc_veto_examples.py        (from any folder; prints; BER_SETS overrides the sets folder)
"""
import os

import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_width_chars(250)
SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sets"))
r = pl.read_parquet(os.path.join(SETS, "recall_add_set.parquet"))
print(r.group_by("grp").agg(n=pl.len(), est=pl.col("est").mean()).sort("n", descending=True))
d = pl.read_parquet(os.path.join(SETS, "desc_veto_set.parquet"))
print(d.group_by("kind").len(), d.group_by("num").len())
low = lambda c: pl.col(c).str.to_lowercase()  # noqa: E731
m = d.filter((low("qn").str.contains(r"\bclub\b") & low("sn").str.contains(r"\becole\b"))
             | (low("sn").str.contains(r"\bclub\b") & low("qn").str.contains(r"\becole\b")))
print("club <-> ecole pairs in the descriptor veto:", m.height)
print(m.head(8).select("q", "s", "num", "kind", "p2", "p3", "qn", "sn"))
