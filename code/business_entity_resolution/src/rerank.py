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

OUT = os.environ.get("BER_CE_DIR", os.path.join(WORK, "ce"))          # rows/scores for one base model
# Two transformers, cross-fitted on the two S1 halves: side "a" trains on group "train" and scores group "eval";
# side "b" trains on group "eval" and scores group "train". Stage 3 then learns from both groups, and test
# uses the average of both sides' stage-3 probabilities.
SIDE = os.environ.get("BER_CE_SIDE", "a")
SFX = "" if SIDE == "a" else "_b"
TRAIN_GRP, SCORE_GRP = ("train", "eval") if SIDE == "a" else ("eval", "train")
FT = os.environ.get("BER_CE_MODEL", os.path.join(WORK, "ce", "model" + SFX))  # the trained transformer (shared)
# BER_CE_REUSE=<another BER_CE_DIR>: reuse that folder's scores of the same transformer for pairs already scored
REUSE = os.environ.get("BER_CE_REUSE", "")
LO, HI, P1_2ND = 0.01, 0.995, 0.2
MAXLEN = 128
LIMIT = int(os.environ.get("BER_CE_LIMIT", "0"))      # smoke test: > 0 = only this many rows per step
DEV = "cuda"


def texts(split, ids):
    """int id -> 'lowercased name | address' for the S1/S2/S3 rows of a split whose id is in `ids` (column id).
    One file at a time, keeping only the needed rows, so memory stays low."""
    out = []
    for k in (1, 2, 3):
        out.append(pl.scan_parquet(os.path.join(WORK, f"{split}_s{k}.parquet")).select(
            "entity_id", "business_name", "business_address").with_columns(id=id_to_int("entity_id"))
            .join(ids.lazy(), on="id").select(
            "id", t=(pl.col("business_name").fill_null("").str.to_lowercase() + " | " +
                     pl.col("business_address").fill_null("").str.to_lowercase()).str.slice(0, 160)).collect())
    return pl.concat(out)


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
    tx = texts(split, pl.concat([rows.select(id=pl.col("q").cast(pl.Int64)), rows.select(id=pl.col("s").cast(pl.Int64))]).unique())
    rows = rows.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    r = (rows.join(tx.rename({"id": "q", "t": "tq"}), on="q", how="left")
             .join(tx.rename({"id": "s", "t": "ts"}), on="s", how="left"))
    return r.with_columns(pl.col("tq").fill_null(""), pl.col("ts").fill_null(""))


def train(base="microsoft/mdeberta-v3-base", bs=64, lr=2e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = pl.read_parquet(os.path.join(OUT, "train_rows.parquet")).filter(pl.col("grp") == TRAIN_GRP)
    rows = rows.sample(fraction=1.0, shuffle=True, seed=0)
    if LIMIT:
        rows = rows.head(LIMIT)
    r = _pairs(rows, "train")
    log(f"transformer side {SIDE}: training rows {r.height} (group {TRAIN_GRP}), positives {r['label'].sum()} -> {FT}")
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
    # batches of similar text length (less padding, faster): within blocks of 64 batches of the shuffled rows,
    # sort by length, cut into batches, shuffle the batch order; fixed seed, so a resumed run sees the same order
    lens = np.array([len(a) + len(b) for a, b in zip(tq, ts)])
    order, rng, blk = [], np.random.default_rng(1), bs * 64
    for a in range(0, len(lens), blk):
        idx = a + np.argsort(lens[a:a + blk], kind="stable")
        bat = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        order += [bat[j] for j in rng.permutation(len(bat))]
    order = np.concatenate(order)
    ck = FT.rstrip("/\\") + "_ckpt.pt"               # resume after a crash (the 4 GB GPU is shared)
    start = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)   # CPU first: loading onto the GPU doubles memory
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        scaler.load_state_dict(st["scaler"]); start = st["step"] + 1
        log(f"resumed from step {st['step']}")
        del st
    m.train()
    for i in range(start, steps):
        ix = order[i * bs:(i + 1) * bs]
        enc = tok([tq[j] for j in ix], [ts[j] for j in ix], padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
            logit = m(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(y[ix], device=DEV))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if i % 500 == 0:
            log(f"step {i}/{steps} loss {loss.item():.4f}")
        if i % 1000 == 999:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "scaler": scaler.state_dict(), "step": i}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    os.makedirs(FT, exist_ok=True)
    m.save_pretrained(FT)
    tok.save_pretrained(FT)


class Scorer:
    """The trained transformer; call with lists of texts, returns logits. Batches are formed from texts of
    similar length (less padding), results come back in the input order."""
    def __init__(self, bs=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.bs = torch, bs
        self.tok = AutoTokenizer.from_pretrained(FT)
        self.m = AutoModelForSequenceClassification.from_pretrained(FT).to(DEV).eval()
        self.dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def __call__(self, tq, ts):
        torch, bs = self.torch, self.bs
        order = np.argsort(np.array([len(a) + len(b) for a, b in zip(tq, ts)]), kind="stable")
        out = np.empty(len(tq), np.float32)
        with torch.no_grad():
            for a in range(0, len(tq), bs):
                ix = order[a:a + bs]
                enc = self.tok([tq[j] for j in ix], [ts[j] for j in ix], padding=True, truncation=True,
                               max_length=MAXLEN, return_tensors="pt").to(DEV)
                with torch.autocast("cuda", dtype=self.dt):
                    out[ix] = self.m(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out


def score(chunk=200_000):
    """Scores this side's close calls (train group SCORE_GRP, and all test close calls) in saved chunks, so a
    crash (out of memory) resumes at the last finished chunk. Pairs already scored by the same transformer in
    BER_CE_REUSE's folder are copied instead of scored."""
    sc = None
    for split, flt in (("train", pl.col("grp") == SCORE_GRP), ("test", pl.lit(True))):
        dst = os.path.join(OUT, f"{split}_ce{SFX}{'_smoke' if LIMIT else ''}.parquet")   # smoke test never overwrites
        if os.path.exists(dst) and not LIMIT:
            log(f"{dst} exists, kept")
            continue
        rows = pl.read_parquet(os.path.join(OUT, f"{split}_rows.parquet")).filter(flt).with_columns(
            pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
        if LIMIT:
            rows = rows.head(LIMIT)
        known = []
        src = os.path.join(REUSE, f"{split}_ce{SFX}.parquet") if REUSE else ""
        if src and os.path.exists(src):
            known = [pl.read_parquet(src, columns=["q", "s", "ce"]).with_columns(
                pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(rows.select("q", "s"), on=["q", "s"])]
            log(f"{split}: {known[0].height} of {rows.height} pairs reused from {src}")
        todo = rows.select("q", "s").unique().sort("q", "s")
        if known:
            todo = todo.join(known[0].select("q", "s"), on=["q", "s"], how="anti")
        pdir = os.path.join(OUT, f"parts_{split}{SFX}{'_smoke' if LIMIT else ''}")
        meta = {"n": todo.height, "chunk": chunk, "model": FT}
        mf = os.path.join(pdir, "meta.json")
        if os.path.exists(mf) and json.load(open(mf)) != meta:      # parts of another run: start over
            import shutil
            shutil.rmtree(pdir)
        os.makedirs(pdir, exist_ok=True)
        json.dump(meta, open(mf, "w"))
        done = []
        for i, a in enumerate(range(0, todo.height, chunk)):
            pf = os.path.join(pdir, f"{i:04d}.parquet")
            if not os.path.exists(pf):
                c = _pairs(todo.slice(a, chunk), split)
                sc = sc or Scorer()
                c.select("q", "s").with_columns(ce=pl.Series(sc(c["tq"].to_list(), c["ts"].to_list()))).write_parquet(pf + ".tmp")
                os.replace(pf + ".tmp", pf)
                log(f"  {split}: scored {min(a + chunk, todo.height)}/{todo.height}")
            done.append(pl.read_parquet(pf))
        ce = pl.concat(known + done) if known or done else pl.DataFrame({"q": [], "s": [], "ce": []})
        r = rows.join(ce, on=["q", "s"], how="left")
        assert r["ce"].null_count() == 0, "pairs without a transformer score"
        r.write_parquet(dst)
        log(f"{split}: {r.height} close-call rows -> {dst}")


S3FEATS = ["p1", "p2", "ce", "ce_margin", "ce_rank", "p2_margin", "n_cc"]


def _s3_features(r):
    """Transformer score relative to the other close-call candidate of the same S2/S3 row."""
    r = r.with_columns(n_cc=pl.len().over("q").cast(pl.Float32),
                       ce_rank=pl.col("ce").rank("ordinal", descending=True).over("q").cast(pl.Float32))
    for c in ("ce", "p2"):
        other = pl.when(pl.len().over("q") > 1).then(pl.col(c).sum().over("q") - pl.col(c)).otherwise(None)
        r = r.with_columns((pl.col(c) - other).alias(f"{c}_margin"))
    return r


def _decisions(b, prob, truth, s1s):
    """Macro F0.5 of the threshold rules and expected-F0.5 set rules on probability column `prob`."""
    from pipeline import decide_expf
    top = b.sort(prob, descending=True).unique("q", keep="first")
    out = {}
    for th in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        out[json.dumps({"type": "thr", "t": th, "prob": "p2"})] = macro_f05_df(
            top.filter(pl.col(prob) >= th).select("s", "q"), truth, s1s)
    for floor in (0.3, 0.5):
        for alpha in (1.0, 1.5, 2.0):
            out[json.dumps({"type": "expf", "floor": floor, "alpha": alpha, "prob": "p2"})] = macro_f05_df(
                decide_expf(b, prob, floor, alpha), truth, s1s)
    return out


def stage3():
    """Stage 3 on the transformer scores of every side present (a: eval group, b: train group), 3 folds.
    Held-out macro F0.5 on the eval-half S1 rows (comparable with earlier runs) and on all S1 rows, for each
    decision rule; the best rule on the eval half is saved as rule<sfx>.json (finalize.py format). Test: each
    side's scores go through the stage-3 model and the stage-3 probabilities are averaged."""
    import xgboost as xgb
    params = dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.05, max_depth=6,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.9, seed=7)
    md = json.load(open(os.path.join(OUT, "select.json")))["model_dir"]
    sides = [x for x in os.environ.get("BER_S3_SIDES", "a,b").split(",")
             if os.path.exists(os.path.join(OUT, f"train_ce{'' if x == 'a' else '_' + x}.parquet"))
             and os.path.exists(os.path.join(OUT, f"test_ce{'' if x == 'a' else '_' + x}.parquet"))]
    tagx = "_" + "".join(sides)          # e.g. _a, _ab; never overwrites the files v7 was built from
    log(f"stage 3 on sides {sides} (output suffix '{tagx}')")
    r = _s3_features(pl.concat([pl.read_parquet(os.path.join(OUT, f"train_ce{'' if x == 'a' else '_' + x}.parquet"))
                                .with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)) for x in sides]))
    assert r.select(pl.struct("q", "s").is_duplicated().sum()).item() == 0
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id"]).select(
        s=id_to_int("entity_id"), h=half("entity_id"))
    p3 = np.zeros(r.height, np.float32)
    for f in range(3):
        tr = r.filter(pl.col("fold") != f)
        mdl = xgb.train(params, xgb.DMatrix(tr.select(S3FEATS).to_numpy(), tr["label"].to_numpy()), 300)
        msk = (r["fold"] == f).to_numpy()
        p3[msk] = mdl.predict(xgb.DMatrix(r.filter(pl.Series(msk)).select(S3FEATS).to_numpy()))
    r = r.with_columns(p3=pl.Series(p3))
    # held-out: p2 replaced by p3 on the close-call rows that have an out-of-sample transformer score
    oof = pl.read_parquet(os.path.join(WORK, "models", md, "oof.parquet"), columns=["q", "s", "p2"]).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    touched = r.select("q").unique().with_columns(t=pl.lit(True))
    new = (oof.join(r.select("q", "s", "p3"), on=["q", "s"], how="left").join(touched, on="q", how="left")
              .with_columns(p2n=pl.when(pl.col("t").is_null()).then(pl.col("p2")).otherwise(pl.col("p3").fill_null(0.0))))
    # S1 rows that exist in this model's training setup; eval half = crc32 half < 500
    tag = json.load(open(os.path.join(WORK, "models", md, "result.json")))["tag"]
    allS = s1.join(pl.read_parquet(os.path.join(WORK, "pairs", tag, "s1.parquet"), columns=["s"]).with_columns(
        pl.col("s").cast(pl.Int64)), on="s")
    truth_all = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    res = {}
    for name, ss in (("eval", allS.filter(pl.col("h") < 500).select("s")), ("all", allS.select("s"))):
        tr_ = truth_all.join(ss, on="s")
        g, n_ = _decisions(new, "p2", tr_, ss), _decisions(new, "p2n", tr_, ss)
        res[name] = (g, n_)
        for k in n_:
            log(f"{name} S1 ({ss.height}) {k}: GBDT {g[k]:.5f} -> with transformer {n_[k]:.5f}")
        kg, kn = max(g, key=g.get), max(n_, key=n_.get)
        log(f"BEST {name}: GBDT {g[kg]:.5f} ({kg}) -> with transformer {n_[kn]:.5f} ({kn})")
    g, n_ = res["eval"]
    kg, kn = max(g, key=g.get), max(n_, key=n_.get)
    json.dump({"note": f"stage 3 sides {sides}, base {md}; best rule on eval-half S1 (rerank.py stage3)",
               "x": [], "y": [], "decision": {"default": json.loads(kn)}, "base": g[kg], "best": n_[kn],
               "all_S1": {"base": max(res["all"][0].values()), "best_rule_score": res["all"][1][kn]}},
              open(os.path.join(OUT, f"rule{tagx}.json"), "w"), indent=1)
    mdl = xgb.train(params, xgb.DMatrix(r.select(S3FEATS).to_numpy(), r["label"].to_numpy()), 300)
    mdl.save_model(os.path.join(OUT, f"stage3{tagx}.json"))
    imp = mdl.get_score(importance_type="total_gain")
    log("stage-3 importance:", {S3FEATS[int(k[1:])]: round(v) for k, v in imp.items()})
    # test: p2 replaced by the mean stage-3 probability over sides for close-call rows, other rows unchanged
    t = None
    for x in sides:
        tx = _s3_features(pl.read_parquet(os.path.join(OUT, f"test_ce{'' if x == 'a' else '_' + x}.parquet")).with_columns(
            pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)))
        tx = tx.select("q", "s", **{f"p3_{x}": pl.Series(mdl.predict(xgb.DMatrix(tx.select(S3FEATS).to_numpy())), dtype=pl.Float32)})
        t = tx if t is None else t.join(tx, on=["q", "s"], how="full", coalesce=True)
    t = t.with_columns(p3=pl.mean_horizontal([f"p3_{x}" for x in sides]))
    if len(sides) > 1:
        log("test stage-3 agreement between sides: corr "
            f"{np.corrcoef(t[f'p3_{sides[0]}'].to_numpy(), t[f'p3_{sides[1]}'].to_numpy())[0, 1]:.4f}, "
            f"decisions differing at 0.5: {t.filter((pl.col(f'p3_{sides[0]}') >= 0.5) != (pl.col(f'p3_{sides[1]}') >= 0.5)).height}")
    te = pl.read_parquet(os.path.join(WORK, f"test_scores_{md}.parquet")).with_columns(
        pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
    tt = t.select("q").unique().with_columns(tch=pl.lit(True))
    te = (te.join(t.select("q", "s", "p3"), on=["q", "s"], how="left").join(tt, on="q", how="left")
            .with_columns(p2=pl.when(pl.col("tch").is_null()).then(pl.col("p2"))
                          .otherwise(pl.col("p3").fill_null(0.0)).cast(pl.Float32)).select("q", "s", "p1", "p2"))
    dst = os.path.join(OUT, f"test_scores_ce{tagx}.parquet")
    te.write_parquet(dst)
    log(f"wrote {dst} ({te.height} rows; {tt.height} S2/S3 rows re-scored); rule {kn} -> rule{tagx}.json")


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
