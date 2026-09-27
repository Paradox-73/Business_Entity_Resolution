"""Blend of reranked models (each = a GBDT base model + stage 3 with transformer scores, from rerank.py).

  python blend.py <name> <model_dir>:<ce_dir>:<sfx> [<model_dir>:<ce_dir>:<sfx> ...]
  e.g. python blend.py cons_sib full_cons:ce:_ab full_sib:ce_sib:_ab

Per base, the probability of a (record, S1) pair is the stage-3 probability on its close-call rows (held-out:
<ce_dir>/oof_s3<sfx>.parquet; test: <ce_dir>/test_scores_ce<sfx>.parquet) and the GBDT p2 elsewhere. The blend is
the weighted mean over the bases that score the pair (26 Sep: was 0 for a missing base). Held-out macro F0.5 on the
eval-half S1 rows is printed for each weight and decision rule, next to each base alone; the best is written as
WORK/test_scores_blend_<name>.parquet + WORK/ce/rule_blend_<name>.json (finalize.py format; pass model dir of the
first base).
"""
import json
import os
import sys
import zlib
import polars as pl
from common import WORK, log, read_truth, id_to_int
from rerank import _decisions


_MIXED = None


def mixed_q():
    """Records whose close calls span both S1 halves ("mixed"): no base scored them before the fold models, so
    BER_BLEND_HALVES=1 keeps their GBDT p2 in every base (the protocol of v7ens's 0.98813)."""
    global _MIXED
    if _MIXED is None:
        _MIXED = (pl.read_parquet(os.path.join(WORK, "ce", "train_rows.parquet"), columns=["q", "grp"])
                    .filter(pl.col("grp") == "mixed").select(pl.col("q").cast(pl.Int64)).unique().with_columns(mx=pl.lit(True)))
    return _MIXED


def held_out(md, ce, sfx):
    """Held-out probability of one base: stage-3 p3 for records with close calls (<ce>/oof_s3<sfx>.parquet),
    GBDT p2 for the other records (WORK/models/<md>/oof.parquet). Returns q, s, p."""
    oof = pl.read_parquet(os.path.join(WORK, "models", md, "oof.parquet"), columns=["q", "s", "p2"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    s3 = pl.read_parquet(os.path.join(WORK, ce, f"oof_s3{sfx}.parquet"), columns=["q", "s", "p3"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    touched = s3.select("q").unique().with_columns(t=pl.lit(True))
    if os.environ.get("BER_BLEND_HALVES") == "1":
        touched = touched.join(mixed_q(), on="q", how="anti")
        s3 = s3.join(mixed_q(), on="q", how="anti")
    d = oof.join(s3, on=["q", "s"], how="full", coalesce=True).join(touched, on="q", how="left")
    return d.select("q", "s", p=pl.when(pl.col("t").is_null()).then(pl.col("p2").fill_null(0.0)).otherwise(pl.col("p3").fill_null(0.0)))


def combine(frames, w):
    """Weighted mean p2 of the bases' p per pair; a pair missing from a base is averaged over the bases that
    have it."""
    out = None
    for i, f in enumerate(frames):
        f = f.rename({"p": f"p{i}"})
        out = f if out is None else out.join(f, on=["q", "s"], how="full", coalesce=True)
    # a pair missing in a base (outside its candidate rows) gets the weighted mean of the bases that have it
    num = sum(w[i] * pl.col(f"p{i}").fill_null(0.0) for i in range(len(frames)))
    den = sum(pl.when(pl.col(f"p{i}").is_null()).then(0.0).otherwise(w[i]) for i in range(len(frames)))
    return out.with_columns(p2=(num / den).cast(pl.Float32))


def main(name, specs):
    """Held-out macro F0.5 of each weight (0.3 / 0.7, 0.5 / 0.5, 0.7 / 0.3 for two bases) and decision rule; the
    best blend of the test scores -> WORK/test_scores_blend_<name>.parquet, its rule -> WORK/ce/rule_blend_<name>.json."""
    specs = [x.split(":") for x in specs]
    frames = [held_out(*x) for x in specs]
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]).select(
        s=id_to_int("entity_id"), h=pl.col("entity_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64))
    tag = json.load(open(os.path.join(WORK, "models", specs[0][0], "result.json")))["tag"]
    ev = s1.filter(pl.col("h") < 500).join(pl.read_parquet(os.path.join(WORK, "pairs", tag, "s1.parquet"), columns=["s"])
                                            .with_columns(pl.col("s").cast(pl.Int64)), on="s").select("s")
    truth = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id")).join(ev, on="s")
    weights = [[a, 1 - a] for a in (0.3, 0.5, 0.7)] if len(frames) == 2 else [[1 / len(frames)] * len(frames)]
    best = (-1, None, None)
    for w in weights:
        res = _decisions(combine(frames, w), "p2", truth, ev)
        k = max(res, key=res.get)
        log(f"weights {w}: best {res[k]:.5f} with {k}")
        if res[k] > best[0] + 1e-6:
            best = (res[k], w, k)
    singles = {}
    for i in range(len(frames)):
        res = _decisions(frames[i].rename({"p": "p2"}), "p2", truth, ev)
        singles[i] = max(res.values())
    sc, w, k = best
    log(f"BEST blend {w}: {sc:.5f} ({k}); single bases: " + ", ".join(f"{specs[i][0]} {v:.5f}" for i, v in singles.items()))
    tests = [pl.read_parquet(os.path.join(WORK, ce, f"test_scores_ce{sfx}.parquet"), columns=["q", "s", "p1", "p2"])
             .with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)) for _, ce, sfx in specs]
    p1 = pl.concat([t.select("q", "s", "p1") for t in tests]).group_by("q", "s").agg(pl.col("p1").max())
    te = combine([t.select("q", "s", p="p2") for t in tests], w).join(p1, on=["q", "s"], how="left").select("q", "s", "p1", "p2")
    te.write_parquet(os.path.join(WORK, f"test_scores_blend_{name}.parquet"))
    json.dump({"note": f"blend {specs} weights {w} (blend.py)", "x": [], "y": [], "decision": {"default": json.loads(k)},
               "base": max(singles.values()), "best": sc}, open(os.path.join(WORK, "ce", f"rule_blend_{name}.json"), "w"), indent=1)
    log(f"wrote test_scores_blend_{name}.parquet ({te.height} rows) and ce/rule_blend_{name}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
