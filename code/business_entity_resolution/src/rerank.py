"""Cross-encoder reranker for close calls (AWS GPU job; smoke-testable on the 4 GB laptop with BER_CE_LIMIT).

A transformer reads the two records' text together ("name | address" of the S2/S3 row and of the S1 row) and
outputs one match score. It only scores "close calls": the top-2 candidates of rows whose decision is not
already certain (best p2 in [LO, HI], or a 2nd candidate with p1 >= P1_2ND). A small stage-3 XGBoost then
combines the GBDT probabilities with the transformer score.

Leakage rule: a train S2/S3 row is "train" if ALL its close-call candidates are S1 rows of the non-eval half
(crc32(s1_id) % 1000 >= 500), "eval" if all are in the eval half, "mixed" otherwise. The transformer trains on
"train" rows; stage 3 trains/validates (3 folds, pipeline folds) on "eval" rows; "mixed" rows keep their p2.

  python rerank.py select <model_dir>          # close calls of train OOF + test -> WORK/ce/{train,test}_rows.parquet
  python rerank.py train [base]                # default microsoft/mdeberta-v3-base (MIT); ~1 h on A10G
  python rerank.py score                       # transformer score for eval-half + test close calls
  python rerank.py stage3                      # OOF macro F0.5 before/after on eval S1 + test scores with p2 replaced
Inputs: WORK/{train,test}_s{1,2,3}.parquet, WORK/models/<model_dir>/oof.parquet, WORK/test_scores_<model_dir>.parquet
Outputs: WORK/ce/test_scores_ce.parquet (q, s, p1, p2) -> finalize.py
"""
import json
import os
import sys
import zlib
import numpy as np
import polars as pl
from common import WORK, log, read_truth, id_to_int, macro_f05_df

OUT = os.path.join(WORK, "ce")
FT = os.path.join(OUT, "model")
LO, HI, P1_2ND = 0.01, 0.995, 0.2
MAXLEN = 128
LIMIT = int(os.environ.get("BER_CE_LIMIT", "0"))      # smoke test: > 0 = only this many rows per step
DEV = "cuda"


def texts(split):
    """int id -> 'lowercased name | address' for all S1/S2/S3 rows of a split."""
    return pl.concat([pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"), columns=["entity_id", "business_name", "business_address"])
                      for k in (1, 2, 3)]).select(
        id=id_to_int("entity_id"),
        t=(pl.col("business_name").fill_null("").str.to_lowercase() + " | " +
           pl.col("business_address").fill_null("").str.to_lowercase()).str.slice(0, 160))


def close_calls(b):
    """Top-2 candidates (by p2) of every row whose decision is uncertain."""
    t2 = b.sort("p2", descending=True).group_by("q", maintain_order=True).head(2).with_columns(
        r=pl.int_range(pl.len()).over("q"))
    g = t2.group_by("q").agg(best=pl.col("p2").max(), p1b=pl.col("p1").filter(pl.col("r") == 1).max())
    keep = g.filter(pl.col("best").is_between(LO, HI) | (pl.col("p1b").fill_null(0) >= P1_2ND)).select("q")
    return t2.join(keep, on="q").drop("r")


def half(col):
    return pl.col(col).map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64)


def select(model_dir):
    os.makedirs(OUT, exist_ok=True)
    oof = pl.read_parquet(os.path.join(WORK, "models", model_dir, "oof.parquet"))
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id"), h=half("entity_id"), country="country")
    rows = close_calls(oof).join(s1, on="s")
    rows = rows.with_columns(grp=pl.when(pl.col("h").max().over("q") < 500).then(pl.lit("eval"))
                                   .when(pl.col("h").min().over("q") >= 500).then(pl.lit("train")).otherwise(pl.lit("mixed")))
    log("close-call S2/S3 rows by group:", rows.unique("q").group_by("grp").len().rows())
    rows.write_parquet(os.path.join(OUT, "train_rows.parquet"))
    log(f"train close calls: {rows.height} rows, {rows['q'].n_unique()} S2/S3 rows "
        f"(of {oof['q'].n_unique()}); positives {rows['label'].sum()}; eval half {rows.filter(pl.col('h') < 500).height}")
    te = pl.read_parquet(os.path.join(WORK, f"test_scores_{model_dir}.parquet"))
    trows = close_calls(te)
    trows.write_parquet(os.path.join(OUT, "test_rows.parquet"))
    log(f"test close calls: {trows.height} rows, {trows['q'].n_unique()} S2/S3 rows (of {te['q'].n_unique()})")
    json.dump({"model_dir": model_dir}, open(os.path.join(OUT, "select.json"), "w"))


def _pairs(rows, split):
    tx = texts(split)
    r = (rows.join(tx.rename({"id": "q", "t": "tq"}), on="q", how="left")
             .join(tx.rename({"id": "s", "t": "ts"}), on="s", how="left"))
    return r.with_columns(pl.col("tq").fill_null(""), pl.col("ts").fill_null(""))


def train(base="microsoft/mdeberta-v3-base", bs=64, lr=2e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = pl.read_parquet(os.path.join(OUT, "train_rows.parquet")).filter(pl.col("grp") == "train")
    rows = rows.sample(fraction=1.0, shuffle=True, seed=0)
    if LIMIT:
        rows = rows.head(LIMIT)
    r = _pairs(rows, "train")
    log(f"transformer training rows {r.height}, positives {r['label'].sum()}")
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(DEV)
    small = torch.cuda.get_device_properties(0).total_memory < 8e9
    if small:      # 4 GB laptop GPU: freeze the word table (most of a multilingual model's weights), checkpoint
        m.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
        m.gradient_checkpointing_enable()
        bs, lr = min(bs, 32), max(lr, 3e-5)
    bf16 = torch.cuda.is_bf16_supported()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = max(1, r.height // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler(enabled=not bf16)
    tq, ts, y = r["tq"].to_list(), r["ts"].to_list(), r["label"].cast(pl.Float32).to_numpy()
    m.train()
    for i in range(steps):
        a, b = i * bs, (i + 1) * bs
        enc = tok(tq[a:b], ts[a:b], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
            logit = m(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(y[a:b], device=DEV))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if i % 500 == 0:
            log(f"step {i}/{steps} loss {loss.item():.4f}")
    os.makedirs(FT, exist_ok=True)
    m.save_pretrained(FT)
    tok.save_pretrained(FT)


def _score(r, bs=256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FT)
    m = AutoModelForSequenceClassification.from_pretrained(FT).to(DEV).eval()
    bf16 = torch.cuda.is_bf16_supported()
    tq, ts = r["tq"].to_list(), r["ts"].to_list()
    out = np.empty(len(tq), np.float32)
    with torch.no_grad():
        for a in range(0, len(tq), bs):
            enc = tok(tq[a:a + bs], ts[a:a + bs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
                out[a:a + bs] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
            if a % (bs * 400) == 0:
                log(f"  scored {a}/{len(tq)}")
    return out


def score():
    for split, flt in (("train", pl.col("grp") == "eval"), ("test", pl.lit(True))):
        rows = pl.read_parquet(os.path.join(OUT, f"{split}_rows.parquet")).filter(flt)
        if LIMIT:
            rows = rows.head(LIMIT)
        r = _pairs(rows, split)
        r = r.with_columns(ce=pl.Series(_score(r))).drop("tq", "ts")
        r.write_parquet(os.path.join(OUT, f"{split}_ce.parquet"))
        log(f"{split}: {r.height} close-call rows scored")


S3FEATS = ["p1", "p2", "ce", "ce_margin", "ce_rank", "p2_margin", "n_cc"]


def _s3_features(r):
    """Transformer score relative to the other close-call candidate of the same S2/S3 row."""
    r = r.with_columns(n_cc=pl.len().over("q").cast(pl.Float32),
                       ce_rank=pl.col("ce").rank("ordinal", descending=True).over("q").cast(pl.Float32))
    for c in ("ce", "p2"):
        other = pl.when(pl.len().over("q") > 1).then(pl.col(c).sum().over("q") - pl.col(c)).otherwise(None)
        r = r.with_columns((pl.col(c) - other).alias(f"{c}_margin"))
    return r


def stage3():
    import xgboost as xgb
    params = dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.05, max_depth=6,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.9, seed=7)
    md = json.load(open(os.path.join(OUT, "select.json")))["model_dir"]
    r = _s3_features(pl.read_parquet(os.path.join(OUT, "train_ce.parquet")))
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]).select(
        s=id_to_int("entity_id"), h=half("entity_id"))
    p3 = np.zeros(r.height, np.float32)
    for f in range(3):
        tr = r.filter(pl.col("fold") != f)
        mdl = xgb.train(params, xgb.DMatrix(tr.select(S3FEATS).to_numpy(), tr["label"].to_numpy()), 300)
        msk = (r["fold"] == f).to_numpy()
        p3[msk] = mdl.predict(xgb.DMatrix(r.filter(pl.Series(msk)).select(S3FEATS).to_numpy()))
    r = r.with_columns(p3=pl.Series(p3))
    # eval: replace p2 by p3 on close-call rows of eval-half S1, compare decisions on eval S1
    oof = pl.read_parquet(os.path.join(WORK, "models", md, "oof.parquet"))
    new = oof.join(r.select("q", "s", "p3"), on=["q", "s"], how="left").with_columns(
        p2n=pl.coalesce("p3", "p2"))
    # eval S1 = eval half of the S1 rows that exist in this model's training setup (e.g. test-like keeps 50% of US)
    tag = json.load(open(os.path.join(WORK, "models", md, "result.json")))["tag"]
    ev = (s1.filter(pl.col("h") < 500).select("s")
            .join(pl.read_parquet(os.path.join(WORK, "pairs", tag, "s1.parquet"), columns=["s"]), on="s"))
    truth = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id")).join(ev, on="s")
    touched = r.select("q").unique()
    new = new.join(touched.with_columns(t=pl.lit(True)), on="q", how="left").with_columns(
        p2n=pl.when(pl.col("t").is_null()).then(pl.col("p2")).otherwise(pl.col("p2n").fill_null(0.0)))
    for th in (0.45, 0.5, 0.55, 0.6, 0.65):
        a = new.sort("p2", descending=True).unique("q", keep="first").filter(pl.col("p2") >= th).select("s", "q")
        b = new.sort("p2n", descending=True).unique("q", keep="first").filter(pl.col("p2n") >= th).select("s", "q")
        log(f"thr {th}: eval S1 macro F0.5 GBDT {macro_f05_df(a, truth, ev):.5f} -> with transformer {macro_f05_df(b, truth, ev):.5f}")
    mdl = xgb.train(params, xgb.DMatrix(r.select(S3FEATS).to_numpy(), r["label"].to_numpy()), 300)
    mdl.save_model(os.path.join(OUT, "stage3.json"))
    imp = mdl.get_score(importance_type="total_gain")
    log("stage-3 importance:", {S3FEATS[int(k[1:])]: round(v) for k, v in imp.items()})
    # test: p2 replaced by p3 for close-call rows (both candidates), other rows unchanged
    t = _s3_features(pl.read_parquet(os.path.join(OUT, "test_ce.parquet")))
    t = t.with_columns(p3=pl.Series(mdl.predict(xgb.DMatrix(t.select(S3FEATS).to_numpy())), dtype=pl.Float32))
    te = pl.read_parquet(os.path.join(WORK, f"test_scores_{md}.parquet"))
    tt = t.select("q").unique().with_columns(tch=pl.lit(True))
    te = (te.join(t.select("q", "s", "p3"), on=["q", "s"], how="left").join(tt, on="q", how="left")
            .with_columns(p2=pl.when(pl.col("tch").is_null()).then(pl.col("p2"))
                          .otherwise(pl.col("p3").fill_null(0.0))).select("q", "s", "p1", "p2"))
    te.write_parquet(os.path.join(OUT, "test_scores_ce.parquet"))
    log(f"wrote {OUT}/test_scores_ce.parquet ({te.height} rows; {tt.height} S2/S3 rows re-scored)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "select":
        select(sys.argv[2])
    elif cmd == "train":
        train(*(sys.argv[2:3]))
    elif cmd == "score":
        score()
    elif cmd == "stage3":
        stage3()
