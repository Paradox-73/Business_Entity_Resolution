"""Recall ceilings on the labelled held-out half, stage by stage.

Held-out half = train S1 rows with zlib.crc32(s1_id) % 1000 < 500 (the evaluation half used by pipeline.py and
rerank.py). Every train record was searched against ALL train S1 rows of its country (work/pairs/full), so the
density matches test. Recall = share of the true (S1, record) pairs of held-out S1 rows found in a candidate set.

Candidate sets (all built out-of-fold on train):
  raw      every pair the four searches returned (work/pairs/full, production blocking)
  stage2   each record's stage-1 top 2 (2nd only if p1 >= 0.01) = stage-2 input (work/models/full_cons/oof.parquet)
  +r35     + stage-1 ranks 3-5 of close-call records, rescored by the transformers (work/ce_x/train_rows.parquet)
  +second  + the second pipeline's stage-2 input (BER_V8=1 blocking; work/gathik/v9/models_full_xgb_cons_oof.parquet)
Methodology section 3 ("Recall ceilings on the held-out half"); root README "Candidate generation".
Reads BER_WORK only (streams the 319M train pairs); writes $BER_SCRATCH/measure/heldout_recall.json (default
C:/ber_scratch).  python heldout_recall.py
"""
import glob
import json
import os
import sys
import zlib
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
from common import WORK, read_truth, id_to_int, log

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
os.makedirs(SCR, exist_ok=True)
PF = os.path.join(WORK, "pairs", "full")
i64 = lambda d: d.select(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))

s1 = pl.read_parquet(os.path.join(PF, "s1.parquet")).with_columns(
    pl.col("s").cast(pl.Int64), h=pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
ev = s1.filter(pl.col("h") < 500).select("s", "country")
qa = pl.read_parquet(os.path.join(PF, "q.parquet")).select(pl.col("q").cast(pl.Int64), "addr_missing", "name_nonlatin")
t = read_truth().select(s=id_to_int("s1_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64))
te = t.join(ev, on="s").join(qa, on="q", how="left")
log(f"held-out S1 rows {ev.height} (of {s1.height}); their true pairs {te.height}; by country "
    f"{dict(te.group_by('country').len().iter_rows())}; record not in pairs/full/q.parquet: {te.filter(pl.col('addr_missing').is_null()).height}")
n_eval_s1 = dict(ev.group_by("country").len().iter_rows())
n_rec = qa.height

# ---- raw shortlist: stream the pair files (q, s only)
found, raw_cnt = [], {}
for f in sorted(glob.glob(os.path.join(PF, "*_[0-9][0-9][0-9].parquet"))):
    p = i64(pl.read_parquet(f, columns=["q", "s"]))
    c = os.path.basename(f).split("_")[0]
    raw_cnt[c] = raw_cnt.get(c, 0) + p.join(ev.select("s"), on="s").height   # pairs whose S1 is held-out
    found.append(p.join(te.select("q", "s"), on=["q", "s"]))
raw = pl.concat(found).unique()
del found
log(f"raw shortlist: {raw.height} held-out true pairs found")

stage2 = i64(pl.read_parquet(os.path.join(WORK, "models", "full_cons", "oof.parquet"), columns=["q", "s"])).unique()
r35 = i64(pl.read_parquet(os.path.join(WORK, "ce_x", "train_rows.parquet"), columns=["q", "s"])).unique()
sec = i64(pl.read_parquet(os.path.join(WORK, "gathik", "v9", "models_full_xgb_cons_oof.parquet"), columns=["q", "s"])).unique()
sets = {"raw": raw, "stage2": stage2, "stage2+r35": pl.concat([stage2, r35]).unique(),
        "second_stage2": sec, "stage2+r35+second": pl.concat([stage2, r35, sec]).unique()}


def size_on_eval(d):
    """pairs of the set whose S1 is held-out, per held-out S1 row, per country"""
    x = d.join(ev, on="s")
    return {c: round(n / n_eval_s1[c], 3) for c, n in x.group_by("country").len().iter_rows()}


res = {"heldout_s1": n_eval_s1, "heldout_true_pairs": dict(te.group_by("country").len().iter_rows()),
       "heldout_true_pairs_no_address": dict(te.filter(pl.col("addr_missing")).group_by("country").len().iter_rows()),
       "raw_pairs_per_heldout_s1": {c: round(raw_cnt[c] / n_eval_s1[c], 3) for c in raw_cnt}}
seg = {"all": pl.lit(True), "with_address": ~pl.col("addr_missing"), "no_address": pl.col("addr_missing"),
       "latin_name": ~pl.col("name_nonlatin"), "non_latin_name": pl.col("name_nonlatin")}
rec = {}
for k, d in sets.items():
    hit = te.join(d.with_columns(hit=pl.lit(True)), on=["q", "s"], how="left").with_columns(pl.col("hit").fill_null(False))
    r = {}
    for sn, cond in seg.items():
        h = hit.filter(cond)
        r[sn] = {"all": round(h["hit"].mean(), 5), "n": h.height, "missed": int((~h["hit"]).sum())}
        for c in ("US", "India"):
            hc = h.filter(pl.col("country") == c)
            r[sn][c] = round(hc["hit"].mean(), 5) if hc.height else None
    # S1 rows whose every true record is in the set
    s1full = hit.group_by("s", "country").agg(pl.col("hit").all())
    r["s1_rows_fully_covered"] = {"all": round(s1full["hit"].mean(), 5)} | {
        c: round(s1full.filter(pl.col("country") == c)["hit"].mean(), 5) for c in ("US", "India")}
    if k != "raw":
        r["pairs_per_heldout_s1"] = size_on_eval(d)
    rec[k] = r
    log(k, json.dumps(r))
res["recall"] = rec
json.dump(res, open(os.path.join(SCR, "heldout_recall.json"), "w"), indent=1)
log("wrote", os.path.join(SCR, "heldout_recall.json"))
