"""v8 submission build, streamed for a machine whose disk cannot hold full pair-feature files (docs/SERVER_RUN.md).

Run with BER_V8=1 (audited blocking and cleaning, candidates.py / normalize.py) after prep.py and embed.py encode_all.
Steps (each skips when its output exists, so the command can simply be re-run after a crash):
  train  a sample (BER_V8_RATE) of the train records pipeline.py may train on (true S1 in the eval half, or no match),
         all countries; pairs + the 57 features at full S1 density -> WORK/v8/train_<country>.parquet
  gbdt   stage-1 XGBoost, 3 folds by true S1 (records without a match by id, as pipeline.py) -> fold models + OOF top-5
         per record -> WORK/v8/train_top.parquet
  test   every test record, chunk by chunk: pairs + features -> mean p1 of the fold models -> top-5 per record
         -> WORK/v8/test_top/<country>_<i>.parquet (features dropped after scoring)
  ce     close calls (best p1 in [0.01, 0.995] or 2nd p1 >= 0.2; their top-2 candidates): 3 cross-encoders
         (BAAI/bge-reranker-v2-m3, fully fine-tuned) each trained on the other folds' train close calls, scoring its
         own fold (out-of-fold) and all test close calls -> WORK/v8/ce_{train,test}_f<k>.parquet
  final  stack XGBoost on [p1, logit, margins, ranks] (3 folds, out-of-fold on train); close-call records take the
         stack probability (their other candidates 0), others keep p1; France: the transformer may only lower p
         (min rule) + legal-form veto (both validated on the LB with v7c-v7i); decision = expected-F0.5 set selection,
         floor 0.5 (v7ens's rule). Writes OUT/v8a (France with the transformer's removals) and OUT/v8b (France without
         the transformer), each matching_results.tsv checked against every rule of the problem statement.

  python v8.py [train|gbdt|test|ce|final|all]
"""
import glob
import json
import os
import sys
import time
import zlib
import numpy as np
import polars as pl
import torch
from common import WORK, OUT, DATA, log, read_truth, id_to_int, int_to_id, macro_f05_df
from candidates import CountryIndex, FEATURES, Q_COLS, CHUNK
from pipeline import decide_expf

V8 = os.path.join(WORK, "v8" + os.environ.get("BER_V8_TAG", ""))
RATE = float(os.environ.get("BER_V8_RATE", "0.08"))
SMOKE = os.environ.get("BER_V8_SMOKE") == "1"          # tiny run: 1 test chunk per country, few rounds / steps
MODEL = os.environ.get("BER_V8_MODEL", "BAAI/bge-reranker-v2-m3")
CE_TRAIN = int(os.environ.get("BER_V8_CE_TRAIN", "300000"))
ROUNDS = 20 if SMOKE else 600
K = 5
LO, HI, P1_2ND = 0.01, 0.995, 0.2
XP = dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.06, max_depth=10, min_child_weight=20,
          subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=7)
FR_FORMS = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"]


def crc(col):
    return pl.col(col).map_elements(lambda x: zlib.crc32(x.encode()), return_dtype=pl.Int64)


def load(split, k, cols=None):
    return pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"), columns=cols)


def pairs_frame(p):
    """chunk_pairs output -> ints q, s + float32 features."""
    return p.select(q=id_to_int("entity_id"), s=id_to_int("entity_id_s"),
                    *[pl.col(f).cast(pl.Float32) for f in FEATURES])


def top_k(df, prob, k=K):
    return df.sort(prob, descending=True).group_by("q", maintain_order=True).head(k)


# ---------------------------------------------------------------- train sample
def step_train():
    os.makedirs(V8, exist_ok=True)
    truth = read_truth()
    for c in ("US", "India"):
        out = os.path.join(V8, f"train_{c}.parquet")
        if os.path.exists(out):
            continue
        s1 = load("train", 1).filter(pl.col("country") == c)
        q = pl.concat([load("train", k).filter(pl.col("country") == c) for k in (2, 3)])
        q = q.join(truth.rename({"q_id": "entity_id"}), on="entity_id", how="left")
        q = q.with_columns(h=crc("entity_id"), hs=crc("s1_id"))
        rate = 0.002 if SMOKE else RATE
        q = q.filter((pl.col("s1_id").is_null() | (pl.col("hs") % 1000 < 500)) & (pl.col("h") % 100000 < rate * 100000))
        log(f"train {c}: {q.height} sampled records (no match {q['s1_id'].null_count()})")
        idx = CountryIndex(s1, "train")
        parts = [pairs_frame(idx.chunk_pairs(q.select(Q_COLS).slice(a, CHUNK))) for a in range(0, q.height, CHUNK)]
        del idx
        torch.cuda.empty_cache()
        lab = q.select(q=id_to_int("entity_id"), true_s=id_to_int("s1_id"), sid="s1_id", qid="entity_id")
        p = pl.concat(parts).join(lab, on="q", how="left").with_columns(
            label=(pl.col("s") == pl.col("true_s")).fill_null(False), country=pl.lit(c))
        p = p.with_columns(fold=pl.when(pl.col("sid").is_not_null()).then(crc("sid") % 3)
                           .otherwise(pl.col("q") % 3).cast(pl.Int8)).drop("sid", "qid")
        p.write_parquet(out)
        log(f"train {c}: {p.height} pairs, positives {int(p['label'].sum())} -> {out}")


# ---------------------------------------------------------------- stage-1 GBDT
def step_gbdt():
    import xgboost as xgb
    out = os.path.join(V8, "train_top.parquet")
    if os.path.exists(out):
        return
    P = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(V8, "train_*.parquet")))
                   if not f.endswith("_top.parquet")])
    X = P.select(FEATURES).to_numpy().astype(np.float32)
    y = P["label"].cast(pl.Float32).to_numpy()
    fold = P["fold"].to_numpy()
    P = P.select("q", "s", "true_s", "label", "fold", "country")
    p1 = np.zeros(len(y), np.float32)
    for f in range(3):
        tr = fold != f
        m = xgb.train(XP, xgb.QuantileDMatrix(X[tr], y[tr]), ROUNDS)
        m.save_model(os.path.join(V8, f"gbdt_f{f}.json"))
        p1[~tr] = m.predict(xgb.DMatrix(X[~tr]))
        log(f"gbdt fold {f}: trained on {int(tr.sum())} pairs")
    del X
    top = top_k(P.with_columns(p1=pl.Series(p1)), "p1")
    top.write_parquet(out)
    log(f"gbdt: OOF top-{K} {top.height} rows; true pairs kept {int(top['label'].sum())}")


# ---------------------------------------------------------------- test, streamed
def step_test():
    import xgboost as xgb
    d = os.path.join(V8, "test_top")
    os.makedirs(d, exist_ok=True)
    models = []
    for f in range(3):
        m = xgb.Booster()
        m.load_model(os.path.join(V8, f"gbdt_f{f}.json"))
        m.set_param({"device": "cuda"})
        models.append(m)
    for c in ("France", "India", "US"):
        if os.path.exists(os.path.join(d, f"{c}.done")):
            continue
        s1 = load("test", 1).filter(pl.col("country") == c)
        q = pl.concat([load("test", k).filter(pl.col("country") == c) for k in (2, 3)]).select(Q_COLS)
        idx = CountryIndex(s1, "test")
        n = q.height if not SMOKE else min(q.height, CHUNK)
        for i, a in enumerate(range(0, n, CHUNK)):
            out = os.path.join(d, f"{c}_{i:03d}.parquet")
            if os.path.exists(out):
                continue
            p = pairs_frame(idx.chunk_pairs(q.slice(a, CHUNK)))
            dm = xgb.DMatrix(p.select(FEATURES).to_numpy().astype(np.float32))
            p1 = np.mean([m.predict(dm) for m in models], axis=0).astype(np.float32)
            t = top_k(p.select("q", "s").with_columns(p1=pl.Series(p1)), "p1")
            t.with_columns(country=pl.lit(c)).write_parquet(out + ".tmp")
            os.replace(out + ".tmp", out)
            log(f"test {c} chunk {i}: {p.height} pairs -> top-{K} {t.height}")
        del idx
        torch.cuda.empty_cache()
        open(os.path.join(d, f"{c}.done"), "w").close()


# ---------------------------------------------------------------- cross-encoder on close calls
def close_calls(b):
    t = b.sort("p1", descending=True).with_columns(r=pl.int_range(pl.len()).over("q"))
    g = t.group_by("q").agg(best=pl.col("p1").max(), second=pl.col("p1").filter(pl.col("r") == 1).max())
    keep = g.filter(pl.col("best").is_between(LO, HI) | (pl.col("second").fill_null(0) >= P1_2ND)).select("q")
    return t.filter(pl.col("r") < 2).join(keep, on="q").drop("r")


def texts(split, ids_q, ids_s):
    """int id -> 'name | address' (lowercase, 160 chars), as rerank.py."""
    out = []
    for k, ids in ((1, ids_s), (2, ids_q), (3, ids_q)):
        out.append(load(split, k, ["entity_id", "business_name", "business_address"]).with_columns(id=id_to_int("entity_id"))
                   .join(ids.to_frame("id"), on="id").select(
            "id", t=(pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
            .str.to_lowercase().str.slice(0, 160)))
    return pl.concat(out).unique("id")


def with_texts(rows, split):
    tx = texts(split, rows["q"].unique(), rows["s"].unique())
    return (rows.join(tx.rename({"id": "q", "t": "tq"}), on="q", how="left")
                .join(tx.rename({"id": "s", "t": "ts"}), on="s", how="left")
                .with_columns(pl.col("tq").fill_null(""), pl.col("ts").fill_null("")))


def step_ce():
    from dl_matcher_exp import CrossEncoder
    import dl_matcher_exp
    dl_matcher_exp.MODEL = MODEL
    tr_cc = close_calls(pl.read_parquet(os.path.join(V8, "train_top.parquet")))
    te_cc = close_calls(pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(V8, "test_top", "*.parquet")))]))
    log(f"close calls: train {tr_cc.height} pairs ({tr_cc['q'].n_unique()} records), test {te_cc.height} pairs "
        f"({te_cc['q'].n_unique()} records)")
    tr_cc, te_cc = with_texts(tr_cc, "train"), with_texts(te_cc, "test")
    for f in range(3):
        o_tr, o_te = os.path.join(V8, f"ce_train_f{f}.parquet"), os.path.join(V8, f"ce_test_f{f}.parquet")
        if os.path.exists(o_tr) and os.path.exists(o_te):
            continue
        tr = tr_cc.filter(pl.col("fold") != f)
        pos, neg = tr.filter("label"), tr.filter(~pl.col("label"))
        cap = 2000 if SMOKE else CE_TRAIN
        n_pos = min(pos.height, cap // 2)
        n_neg = min(neg.height, cap - n_pos)
        tr = pl.concat([pos.sample(n_pos, seed=7), neg.sample(n_neg, seed=7)])
        log(f"ce fold {f}: training on {tr.height} close-call pairs (positives {n_pos})")
        enc = CrossEncoder()
        enc.fit(tr["tq"].to_list(), tr["ts"].to_list(), tr["label"].cast(pl.Float32).to_numpy())
        own = tr_cc.filter(pl.col("fold") == f)
        own.select("q", "s", ce=pl.Series(enc.score(own["tq"].to_list(), own["ts"].to_list()))).write_parquet(o_tr)
        te_cc.select("q", "s", ce=pl.Series(enc.score(te_cc["tq"].to_list(), te_cc["ts"].to_list()))).write_parquet(o_te)
        del enc
        torch.cuda.empty_cache()
        log(f"ce fold {f}: scored {own.height} train (out-of-fold) + {te_cc.height} test pairs")


# ---------------------------------------------------------------- stack, rules, files
def stack_feats(r):
    for c in ("p1", "ce"):
        other = pl.when(pl.len().over("q") > 1).then(pl.col(c).sum().over("q") - pl.col(c)).otherwise(None)
        r = r.with_columns((pl.col(c) - other).alias(f"{c}_margin"),
                           pl.col(c).rank("ordinal", descending=True).over("q").cast(pl.Float32).alias(f"{c}_rank"))
    return r.with_columns(n_cc=pl.len().over("q").cast(pl.Float32))


SF = ["p1", "ce", "p1_margin", "ce_margin", "p1_rank", "ce_rank", "n_cc"]


def fr_veto(b):
    """France legal-form veto (finalize.py): both raw names carry a legal form and none agree -> p = 0."""
    def forms(split_ks):
        d = pl.concat([load("test", k, ["entity_id", "business_name", "country"]) for k in split_ks]).filter(
            pl.col("country") == "France")
        t = (d["business_name"].fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
               .str.replace_all(r"\b([a-z])\.", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
        return d.select(id=id_to_int("entity_id"), f=t.str.split(" ").list.eval(
            pl.element().replace({"5arl": "sarl", "5as": "sas"})).list.eval(
            pl.element().filter(pl.element().is_in(FR_FORMS))).list.unique())
    b = b.join(forms((1,)).rename({"id": "s", "f": "fs"}), on="s", how="left").join(
        forms((2, 3)).rename({"id": "q", "f": "fq"}), on="q", how="left")
    veto = ((pl.col("country") == "France") & (pl.col("fs").list.len() > 0) & (pl.col("fq").list.len() > 0)
            & (pl.col("fs").list.set_intersection("fq").list.len() == 0)).fill_null(False)
    log(f"France legal-form veto: {b.filter(veto).height} pairs set to 0")
    return b.with_columns(p=pl.when(veto).then(0.0).otherwise(pl.col("p"))).drop("fs", "fq")


def final_prob(top, cc, p3):
    """Close-call records: stack probability on their top-2, 0 on other candidates; other records: p1."""
    s = cc.select("q", "s", p3=pl.Series(p3))
    touched = cc.select("q").unique().with_columns(t=pl.lit(True))
    return (top.join(s, on=["q", "s"], how="left").join(touched, on="q", how="left")
               .with_columns(p=pl.when(pl.col("t").is_null()).then(pl.col("p1")).otherwise(pl.col("p3").fill_null(0.0)))
               .drop("t"))


def write_tsv(M, tag):
    s1 = pl.concat([load("test", 1, ["entity_id"])]).select(s=id_to_int("entity_id"), sid="entity_id")
    g = M.group_by("s").agg(pl.col("q").unique().sort()).with_columns(
        matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    out = (s1.join(g.select("s", "matched_entity_ids"), on="s", how="left")
             .with_columns(pl.col("matched_entity_ids").fill_null(""))
             .select(source1_entity_id="sid", matched_entity_ids="matched_entity_ids"))
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    out.write_csv(os.path.join(d, "matching_results.tsv"), separator="\t", quote_style="never")
    # every rule of the problem statement (check_submission.py): one row per test S1, ids exist, no duplicates
    ids = pl.concat([load("test", k, ["entity_id"]) for k in (2, 3)])["entity_id"]
    e = out.with_columns(i=pl.col("matched_entity_ids").str.split(",")).explode("i").filter(pl.col("i") != "")
    bad = {"rows": out.height - s1.height, "dup_s1": out.height - out["source1_entity_id"].n_unique(),
           "dup_id": e.height - e.unique(["source1_entity_id", "i"]).height,
           "unknown_id": e.filter(~pl.col("i").is_in(ids.implode())).height,
           "id_in_two_lists": e.height - e["i"].n_unique()}
    log(f"{tag}: {out.height} S1 rows, {(out['matched_entity_ids'] != '').sum()} non-empty, {e.height} matched ids; "
        f"checks {bad} -> {'PASS' if not any(bad.values()) else 'FAIL'}")
    return out, bad


def step_final():
    import xgboost as xgb
    ttop = pl.read_parquet(os.path.join(V8, "train_top.parquet"))
    tr_cc = close_calls(ttop)
    tr_cc = tr_cc.join(pl.concat([pl.read_parquet(os.path.join(V8, f"ce_train_f{f}.parquet")) for f in range(3)]),
                       on=["q", "s"], how="left")
    assert tr_cc["ce"].null_count() == 0
    tr_cc = stack_feats(tr_cc)
    p3 = np.zeros(tr_cc.height, np.float32)
    sp = dict(XP, max_depth=4, eta=0.05)
    for f in range(3):
        a = tr_cc.filter(pl.col("fold") != f)
        m = xgb.train(sp, xgb.QuantileDMatrix(a.select(SF).to_numpy(), a["label"].cast(pl.Float32).to_numpy()), 300)
        p3[(tr_cc["fold"] == f).to_numpy()] = m.predict(xgb.DMatrix(tr_cc.filter(pl.col("fold") == f).select(SF).to_numpy()))
    mall = xgb.train(sp, xgb.QuantileDMatrix(tr_cc.select(SF).to_numpy(), tr_cc["label"].cast(pl.Float32).to_numpy()), 300)

    # held-out on the train sample (records of the sample, their S1 groups; optimistic, see EXPERIMENTS.md)
    b = final_prob(ttop, tr_cc, p3)
    truth = ttop.filter(pl.col("true_s").is_not_null()).select(s="true_s", q="q").unique()   # incl. pairs outside top-5
    s1e = truth.select("s").unique()
    res = {}
    for name, prob in (("GBDT stage 1", "p1"), ("v8 (stack on close calls)", "p")):
        res[name] = round(macro_f05_df(decide_expf(b, prob, 0.5, 1.0), truth, s1e), 5)
    log(f"train-sample held-out (expF 0.5), S1 groups of sampled records: {res}")

    # test
    top = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(V8, "test_top", "*.parquet")))])
    cc = close_calls(top)
    ces = [pl.read_parquet(os.path.join(V8, f"ce_test_f{f}.parquet")).rename({"ce": f"ce{f}"}) for f in range(3)]
    for c in ces:
        cc = cc.join(c, on=["q", "s"], how="left")
    cc = stack_feats(cc.with_columns(ce=pl.mean_horizontal("ce0", "ce1", "ce2")))
    b = final_prob(top, cc, mall.predict(xgb.DMatrix(cc.select(SF).to_numpy())))
    fr = pl.col("country") == "France"
    summary = {"train_sample_heldout": res, "test_close_call_records": cc["q"].n_unique(), "test_records": top["q"].n_unique()}
    for tag, prob in (("v8a", pl.when(fr).then(pl.min_horizontal("p1", "p")).otherwise(pl.col("p"))),
                      ("v8b", pl.when(fr).then(pl.col("p1")).otherwise(pl.col("p")))):
        bb = fr_veto(b.with_columns(p=prob.cast(pl.Float32)))
        M = decide_expf(bb, "p", 0.5, 1.0)
        out, bad = write_tsv(M, tag + ("_smoke" if SMOKE else ""))
        per = (M.join(bb.select("q", "country").unique("q"), on="q").group_by("country").agg(
            records=pl.len(), s1=pl.col("s").n_unique()).sort("country").rows())
        summary[tag] = {"checks": bad, "matched_records_and_s1_by_country": per}
        log(f"{tag}: matched records / S1 rows by country {per}")
    json.dump(summary, open(os.path.join(V8, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    steps = {"train": step_train, "gbdt": step_gbdt, "test": step_test, "ce": step_ce, "final": step_final}
    todo = list(steps) if len(sys.argv) < 2 or sys.argv[1] == "all" else [sys.argv[1]]
    for s in todo:
        t0 = time.time()
        log(f"=== {s}")
        steps[s]()
        log(f"=== {s} done in {time.time() - t0:.0f}s")
