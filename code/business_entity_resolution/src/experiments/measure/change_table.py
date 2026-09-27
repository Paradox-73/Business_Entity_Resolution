"""Methodology section 2.1, "Noise in true pairs": share of the sampled true pairs with each kind of name and address
change, per country. One pair can carry several changes.

  python change_table.py        (after change_sample.py)

Reads $BER_SCRATCH/measure/ops_true_{US,India}.parquet; writes $BER_SCRATCH/measure/change_table.json.
Name changes are grouped into generator operations by france_fix/artifacts/census/groups.py.
"""
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(SRC, "france_fix", "artifacts", "census"))
import polars as pl  # noqa: E402
from groups import name_groups  # noqa: E402

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
NAME = ["CASE", "LEGAL_DROP", "LEGAL_ADD", "LEGAL_CHANGE", "LEGAL_ABBREV", "LEGAL_DOT", "SWAP", "DROP", "ADD", "TYPO", "LEET",
        "ACCENT", "DOMAIN", "SQUASH", "SCRIPT", "ORDER", "TITLE", "TRADINGAS", "AMP", "ACRONYM", "DBLSPACE", "BRACKET",
        "HYPHEN", "DUP", "IDTAG"]
ADDR = {"missing": r"^a_missing$", "upper": r"^a_upper$", "street_abbr": r"^a_street_abbr$", "state_expand": r"^a_state_expand$",
        "state_abbr": r"^a_state_abbr$", "script": r"^a_script$", "shuffle": r"^a_shuffle$", "drop": r"^a_drop:",
        "add": r"^a_add:", "chg": r"^a_chg:", "typo": r"^a_typo$", "num_missing": r"^a_num_missing$",
        "num_up1_20": r"^a_num_up(1|2_5|6_20)$", "num_down1_20": r"^a_num_down(1|2_5|6_20)$",
        "num_big": r"^a_num_(up|down)big$", "unit_drop": r"^a_unit_drop$", "trunc": r"^a_comp_trunc$",
        "extra_numbers": r"^a_extra_numbers$"}
out = {}
for c in ("US", "India"):
    x = pl.read_parquet(os.path.join(SCR, f"ops_true_{c}.parquet"))
    gl = [name_groups([k for k in o if k.startswith("n_")]) for o in x["ops"].to_list()]
    n = len(gl)
    r = {"none": sum(1 for g in gl if not g) / n}
    for k in NAME:
        r[k] = sum(1 for g in gl if k in g) / n
    for k, pat in ADDR.items():
        r["a_" + k] = x.select(pl.col("ops").list.eval(pl.element().str.contains(pat)).list.any().mean()).item()
    out[c] = r
for k in out["US"]:
    print(f"{k:18s} US {100 * out['US'][k]:6.2f}%  India {100 * out['India'][k]:6.2f}%")
json.dump(out, open(os.path.join(SCR, "change_table.json"), "w"), indent=1)
