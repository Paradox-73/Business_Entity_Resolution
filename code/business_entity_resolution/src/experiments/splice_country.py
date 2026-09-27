"""Submission file whose rows for one country come from another file (lines copied as they are).

  python splice_country.py <main.tsv> <donor.tsv> <out_dir> [country=France]
Rows of S1 entities of <country> are taken from <donor.tsv>, all other rows from <main.tsv> (row order of main).
Use: US/India of a new model with the France rows of a version whose France score is known from the leaderboard,
so LB(out) - LB(donor) = 0.85 x the US/India change only.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import polars as pl
from common import WORK


def lines(p):
    with open(p, encoding="utf8") as f:
        head = f.readline()
        return head, {ln.split("\t", 1)[0]: ln for ln in f}


def main(main_tsv, donor_tsv, out_dir, country="France"):
    ids = set(pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"])
              .filter(pl.col("country") == country)["entity_id"].cast(pl.Utf8).to_list())
    head, m = lines(main_tsv)
    head2, d = lines(donor_tsv)
    assert head == head2 and m.keys() == d.keys(), "the two files must have the same header and S1 rows"
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf8", newline="") as f:
        f.write(head)
        for k, ln in m.items():
            if k in ids:
                n += ln != d[k]
                ln = d[k]
            f.write(ln)
    print(f"{len(ids)} {country} S1 rows taken from {donor_tsv} ({n} differ from {main_tsv}) -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
