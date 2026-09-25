"""Simulate test conditions on out-of-fold predictions and re-tune the decision rule.

Test has ~5.7 S2/S3 rows per S1 vs 4.68 in train, while predicted matches per S1 are about equal,
so test has ~1.9x more unmatched ("decoy") rows per S1. We duplicate unmatched OOF rows (with their
candidates and scores) until decoys per S1 match that rate, then score decision rules.

Usage: python decoy_sim.py <model_dir_name> [decoys_per_s1_test]
"""
import json
import os
import sys
import numpy as np
import polars as pl
from common import WORK, log, macro_f05_df
from pipeline import train_meta, decide_expf


def simulate(oof, qmap, target_decoys_per_s1, n_s1, seed=0):
    matched = qmap.select("q")
    dec = oof.select("q").unique().join(matched, on="q", how="anti")
    cur = dec.height / n_s1
    frac = max(target_decoys_per_s1 - cur, 0) / cur
    extra = []
    rng = np.random.default_rng(seed)
    rep = int(frac)
    rest = frac - rep
    pick = [dec] * rep + [dec.filter(pl.Series(rng.random(dec.height) < rest))]
    for i, d in enumerate(pick, start=1):
        rows = oof.join(d, on="q").with_columns(q=pl.col("q") + i * 100_000_000_000)
        extra.append(rows)
    log(f"decoys per S1: {cur:.2f} -> {target_decoys_per_s1:.2f} (duplicated {sum(e['q'].n_unique() for e in extra)} rows)")
    return pl.concat([oof] + extra)


def main(md_name, target=2.30):
    md = os.path.join(WORK, "models", md_name)
    oof = pl.read_parquet(os.path.join(md, "oof.parquet"))
    tag = "full"
    s1, t, qmap = train_meta(tag)
    s1e = s1.filter("is_eval").select("s")
    te = t.join(s1e, on="s")
    res = json.load(open(os.path.join(md, "result.json")))
    dec = res["decision"]
    base = macro_f05_df(decide_expf(oof, "p2", dec["floor"], dec["alpha"]), te, s1e)
    # decoys per S1 measured over the eval S1 half only (their share of unmatched rows ~ half)
    n_s1 = s1.height
    sim = simulate(oof, qmap, target, n_s1)
    now = macro_f05_df(decide_expf(sim, "p2", dec["floor"], dec["alpha"]), te, s1e)
    log(f"{md_name}: OOF {base:.5f} | with test-like decoy rate, current rule {dec}: {now:.5f}")
    best = (now, dec)
    for floor in (0.3, 0.5, 0.7, 0.8, 0.9):
        for alpha in (1.0, 1.5, 2.0, 3.0, 4.0):
            sc = macro_f05_df(decide_expf(sim, "p2", floor, alpha), te, s1e)
            if sc > best[0]:
                best = (sc, {"type": "expf", "floor": floor, "alpha": alpha})
    log(f"best rule under test-like decoys: {best[1]} -> {best[0]:.5f}; "
        f"same rule on plain OOF: {macro_f05_df(decide_expf(oof, 'p2', best[1]['floor'], best[1]['alpha']), te, s1e):.5f}")
    json.dump({"decision_testlike": best[1], "score_testlike": best[0], "score_testlike_current": now,
               "target_decoys_per_s1": target}, open(os.path.join(md, "decoy_sim.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.30)
