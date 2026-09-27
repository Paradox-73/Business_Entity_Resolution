"""Sample of labelled US/India pairs run through the change detector (france_fix/artifacts/census/ops.py): 80,000 true
pairs and 80,000 decoys per country, each decoy with its best stage-1 candidate (out-of-fold table). Methodology
section 2.1 ("Noise in true pairs" and "Artefact 2") is computed from these samples by change_table.py and
lowercase_test.py.

  python change_sample.py        (from any folder; 6 worker processes, about 10 min on the laptop)

Reads WORK/train_s{1,2,3}.parquet, the train ground truth and WORK/models/full_cons/oof.parquet.
Writes $BER_SCRATCH/measure/ops_{true,decoy}_{US,India}.parquet (q, s, ops = sorted list of detected changes) and
change_sample.json: all-lowercase share split by whether the name has a word-level change (n_swap: / n_add:).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "france_fix", "artifacts", "census"))
import polars as pl  # noqa: E402
from ops import ops  # noqa: E402
from common import WORK, id_to_int, read_truth  # noqa: E402

N = 80000
SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")


def work(rows):
    """Detected changes of each (record name, record address, S1 name, S1 address, country) row."""
    return [sorted(ops(qn, qa, sn, sa, c)) for qn, qa, sn, sa, c in rows]


if __name__ == "__main__":
    os.makedirs(SCR, exist_ok=True)
    t0 = time.time()
    tr = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"),
                         columns=["entity_id", "business_name", "business_address", "country"]).select(
        s=id_to_int("entity_id"), sn="business_name", sa="business_address", country="country")
    rec = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"),
                                     columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).select(
        q=id_to_int("entity_id"), qn="business_name", qa="business_address")
    oof = pl.read_parquet(os.path.join(WORK, "models", "full_cons", "oof.parquet"), columns=["q", "s", "p1"])
    dec = oof.sort("p1", descending=True).unique("q", keep="first").join(tr, on="q", how="anti").select("q", "s")
    out = {}
    for grp, pairs in (("true", tr), ("decoy", dec)):
        d = pairs.join(s1, on="s").join(rec, on="q")
        for c in ("US", "India"):
            x = d.filter(pl.col("country") == c).sample(N, seed=5)
            rows = list(zip(x["qn"].to_list(), x["qa"].to_list(), x["sn"].to_list(), x["sa"].to_list(), x["country"].to_list()))
            ch = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
            with Pool(6) as p:
                res = [r for part in p.map(work, ch) for r in part]
            x = x.with_columns(ops=pl.Series(res, dtype=pl.List(pl.Utf8)))
            x.select("q", "s", "ops").write_parquet(os.path.join(SCR, f"ops_{grp}_{c}.parquet"))
            x = x.with_columns(low=pl.col("ops").list.contains("n_lower"),
                               wchg=pl.col("ops").list.eval(pl.element().str.contains(r"^n_(swap|add):")).list.any(),
                               nup=pl.col("ops").list.eval(pl.element().str.contains(r"^a_num_up")).list.any())
            out[f"{grp}_{c}"] = {"n": x.height, "low_all": x["low"].mean(), "wordchange_n": int(x["wchg"].sum()),
                                 "low_given_wordchange": x.filter("wchg")["low"].mean(),
                                 "low_given_no_wordchange": x.filter(~pl.col("wchg"))["low"].mean(),
                                 "num_up_share": x["nup"].mean()}
            print(grp, c, out[f"{grp}_{c}"], round(time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(SCR, "change_sample.json"), "w"), indent=1)
