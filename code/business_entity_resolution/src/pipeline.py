"""End-to-end pipeline.

  python pipeline.py dev  [frac]   # validate on a slice of TRAIN: blocking recall, OOF macro F0.5, saves model
  python pipeline.py test          # run the saved model on TEST, write output/*.tsv

Decision rule: every S2/S3 row is given to its single highest-probability S1 candidate if that
probability >= threshold (training data shows an S2/S3 row never matches two S1 rows),
otherwise to nobody. The threshold is tuned on out-of-fold predictions with the real metric.
"""
import json
import os
import sys
import zlib
import lightgbm as lgb
import numpy as np
import polars as pl
from common import WORK, OUT, log, read_truth, macro_f05
from candidates import FEATURES, country_pairs

N_FOLDS = 3
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, num_threads=11, verbose=-1)
ROUNDS = 400
TRAIN_ROWS = 6_000_000


def load(split, k, cols=None):
    return pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"), columns=cols)


def Xmat(df):
    return df.select([pl.col(c).cast(pl.Float32) for c in FEATURES]).to_numpy()


def _h(s):
    return zlib.crc32(s.encode()) % N_FOLDS


def decide(pairs, t):
    """pairs: q_id, s_id, prob -> dict s_id -> set(q_id)."""
    best = pairs.sort("prob", descending=True).unique("q_id", keep="first").filter(pl.col("prob") >= t)
    out = {}
    for s, q in best.select("s_id", "q_id").iter_rows():
        out.setdefault(s, set()).add(q)
    return out


def tune(pairs, true_map, s1_ids, grid=np.arange(0.2, 0.95, 0.05)):
    res = [(round(float(t), 2), macro_f05(decide(pairs, t), true_map, s1_ids)) for t in grid]
    return max(res, key=lambda x: x[1]), res


def dev(frac=0.1):
    truth = read_truth()
    s1 = load("train", 1)
    s1 = s1.filter(pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 < frac * 1000,
                                                   return_dtype=pl.Boolean))
    ids = set(s1["entity_id"])
    tr = truth.filter(pl.col("s1_id").is_in(list(ids)))
    matched_all = set(truth["q_id"])
    keep_q = set(tr["q_id"])
    q = pl.concat([load("train", k) for k in (2, 3)])
    q = q.filter(pl.col("entity_id").is_in(list(keep_q)) |
                 (~pl.col("entity_id").is_in(list(matched_all)) &
                  pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 < frac * 1000,
                                                   return_dtype=pl.Boolean)))
    log(f"dev slice: S1 {s1.height}, S2/S3 {q.height} (matched {len(keep_q)})")

    parts = []
    for c in s1["country"].unique().to_list():
        log(f"country {c}")
        country_pairs(s1.filter(pl.col("country") == c), q.filter(pl.col("country") == c), parts.append)
    P = pl.concat(parts).rename({"entity_id": "q_id", "entity_id_s": "s_id"})
    del parts
    tmap = dict(zip(tr["q_id"], tr["s1_id"]))
    qfold = {qq: _h(s) for qq, s in tmap.items()}
    P = P.with_columns(
        label=(pl.col("q_id").replace_strict(tmap, default="") == pl.col("s_id")),
        fold=pl.col("q_id").map_elements(lambda x: qfold.get(x, _h(x)), return_dtype=pl.Int8))
    n_true, n_found = tr.height, int(P["label"].sum())
    log(f"BLOCKING: {P.height} pairs, {P.height / q.height:.1f} per S2/S3 row, "
        f"recall ceiling {n_found / n_true:.4f} ({n_found}/{n_true})")
    for flag in ("from_name", "from_addr"):
        log(f"   recall from {flag} alone: {P.filter(pl.col(flag) & pl.col('label')).height / n_true:.4f}")
    P.write_parquet(os.path.join(WORK, "dev_pairs.parquet"))

    true_map = {}
    for s, qq in tr.iter_rows():
        true_map.setdefault(s, set()).add(qq)
    s1_ids = list(ids)

    # simple rule baseline: best candidate by name+address token-set similarity
    base = P.select("q_id", "s_id", prob=(pl.col("n_tset") + pl.col("a_tset")) / 200)
    (bt, bs), _ = tune(base, true_map, s1_ids)
    log(f"RULE BASELINE macro F0.5 = {bs:.4f} at threshold {bt}")

    oof = np.zeros(P.height, np.float32)
    rng = np.random.default_rng(0)
    for f in range(N_FOLDS):
        trn = P.filter(pl.col("fold") != f)
        if trn.height > TRAIN_ROWS:
            trn = trn.sample(TRAIN_ROWS, seed=f)
        m = lgb.train(PARAMS, lgb.Dataset(Xmat(trn), trn["label"].to_numpy()), ROUNDS)
        mask = (P["fold"] == f).to_numpy()
        oof[mask] = m.predict(Xmat(P.filter(pl.col("fold") == f)))
        log(f"fold {f} done")
    P = P.with_columns(prob=pl.Series(oof))
    (t, s), res = tune(P.select("q_id", "s_id", "prob"), true_map, s1_ids)
    log(f"LIGHTGBM OOF macro F0.5 = {s:.4f} at threshold {t}")
    log("  threshold curve:", res)
    # oracle: perfect model on these candidates (upper bound given blocking)
    orc = P.filter("label").select("q_id", "s_id", prob=pl.lit(1.0))
    log(f"  upper bound with this blocking (perfect model): {macro_f05(decide(orc, 0.5), true_map, s1_ids):.4f}")

    trn = P.sample(min(TRAIN_ROWS, P.height), seed=42)
    m = lgb.train(PARAMS, lgb.Dataset(Xmat(trn), trn["label"].to_numpy()), ROUNDS)
    m.save_model(os.path.join(WORK, "model.txt"))
    imp = sorted(zip(FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    log("top features:", [(a, int(b)) for a, b in imp[:15]])
    json.dump({"threshold": t, "oof_f05": s, "baseline_f05": bs, "frac": frac,
               "recall_ceiling": n_found / n_true, "pairs_per_q": P.height / q.height},
              open(os.path.join(WORK, "dev_result.json"), "w"), indent=1)


def test():
    cfg = json.load(open(os.path.join(WORK, "dev_result.json")))
    m = lgb.Booster(model_file=os.path.join(WORK, "model.txt"))
    s1 = load("test", 1)
    q = pl.concat([load("test", k) for k in (2, 3)])
    os.makedirs(OUT, exist_ok=True)
    match_rows, cand_rows = [], []
    for c in s1["country"].unique().to_list():
        log(f"country {c}")
        saved = os.path.join(WORK, f"test_pairs_{c}.parquet")   # resume: finished countries are not recomputed
        if os.path.exists(saved):
            P = pl.read_parquet(saved)
        else:
            kept = []

            def on_chunk(p):
                kept.append(p.select(q_id="entity_id", s_id="entity_id_s",
                                     prob=pl.Series(m.predict(Xmat(p)), dtype=pl.Float32)))

            country_pairs(s1.filter(pl.col("country") == c), q.filter(pl.col("country") == c), on_chunk)
            P = pl.concat(kept)
            del kept
            P.write_parquet(saved)
        cand_rows.append(P.group_by("s_id").agg(pl.col("q_id").unique().sort().str.join(",")))
        best = (P.sort("prob", descending=True).unique("q_id", keep="first")
                 .filter(pl.col("prob") >= cfg["threshold"]))
        match_rows.append(best.group_by("s_id").agg(pl.col("q_id").sort().str.join(",")))
        log(f"  {c}: {P.height} candidate pairs, {best.height} matches")
        del P, best
    ids = s1.select(source1_entity_id="entity_id")
    for rows, name, col in ((match_rows, "matching_results.tsv", "matched_entity_ids"),
                            (cand_rows, "candidate_pairs.tsv", "candidate_entity_ids")):
        agg = pl.concat(rows).rename({"s_id": "source1_entity_id", "q_id": col})
        out = ids.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
        out.write_csv(os.path.join(OUT, name), separator="\t", quote_style="never")
        log(f"wrote {name}: {out.height} rows, {(out[col] != '').sum()} non-empty")


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev(float(sys.argv[2]) if len(sys.argv) > 2 else 0.1)
    else:
        test()
