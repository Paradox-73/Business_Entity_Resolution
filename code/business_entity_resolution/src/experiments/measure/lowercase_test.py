"""Methodology section 2.1, "Artefact 2": all-lowercase share of sampled true records and decoys whose name adds or
swaps in a noise or descriptor word (word classes of france_fix/artifacts/census/ops.py), and of those whose changed
word is of another class.

  python lowercase_test.py        (after change_sample.py)

Reads $BER_SCRATCH/measure/ops_{true,decoy}_{US,India}.parquet; writes $BER_SCRATCH/measure/lowercase_test.json
(nd_* = noise/descriptor words, other_* = other words; *_low = all-lowercase share).
"""
import json
import os

import polars as pl

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
out = {}
for g in ("true", "decoy"):
    for c in ("US", "India"):
        x = pl.read_parquet(os.path.join(SCR, f"ops_{g}_{c}.parquet")).with_columns(low=pl.col("ops").list.contains("n_lower"))
        m = x.filter(pl.col("ops").list.eval(pl.element().str.contains(r"^n_(swap|add):(noise|desc)$")).list.any())
        o = x.filter(pl.col("ops").list.eval(pl.element().str.contains(r"^n_(swap|add):other$")).list.any())
        out[f"{g}_{c}"] = dict(n=x.height, low_all=round(x["low"].mean(), 4), nd_n=m.height, nd_low=round(m["low"].mean(), 4),
                               nd_low_cnt=int(m["low"].sum()), other_n=o.height, other_low=round(o["low"].mean(), 4))
        print(g, c, out[f"{g}_{c}"])
json.dump(out, open(os.path.join(SCR, "lowercase_test.json"), "w"), indent=1)
