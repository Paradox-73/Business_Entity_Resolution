"""Can deep learning do the final matching? GBDT vs cross-encoder vs both, in memory, at full S1 density.

Sample per country (US, India): N_S1 eval-half train S1 rows (crc32 % 1000 < 500), all their true S2/S3 records, and
unmatched records at the same rate. Candidates come from the audited blocking (candidates.py) searched against ALL
train S1 rows of the country, so competition is at test density. Nothing is written to disk except small results.
Folds: 3 by true S1 (unmatched records by their own id). Every method decides among the SAME top-K candidates per
record (by the GBDT's out-of-fold score) with the same decision rules; score = macro F0.5 over the sampled S1 rows.
  GBDT   XGBoost on the 57 pair features (production stage-1 recipe)
  DL     BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M) fine-tuned end to end on raw "name | address" text pairs
  STACK  small XGBoost on [GBDT p, DL score, their ranks and margins within the record's top-K]

  BER_DLX_S1=40000 BER_DLX_TRAIN=240000 python dl_matcher_exp.py
"""
import json
import os
import time
import zlib
import numpy as np
import polars as pl
import torch
from common import WORK, log, read_truth, id_to_int, macro_f05_df
from candidates import CountryIndex, FEATURES, Q_COLS
from pipeline import decide_expf

N_S1 = int(os.environ.get("BER_DLX_S1", "40000"))
TRAIN_ROWS = int(os.environ.get("BER_DLX_TRAIN", "240000"))
MODEL = os.environ.get("BER_DLX_MODEL", "BAAI/bge-reranker-v2-m3")
K = 5
CH = 125_000
SEED = 7


def crc(col):
    return pl.col(col).map_elements(lambda x: zlib.crc32(x.encode()), return_dtype=pl.Int64)


def build(country, truth):
    """Pairs + features for the sampled S1 rows' records and a same-rate sample of unmatched records."""
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet")).filter(pl.col("country") == country)
    s1 = s1.with_columns(h=crc("entity_id"))
    ev = s1.filter(pl.col("h") % 1000 < 500)
    rate = min(1.0, N_S1 / ev.height)
    samp = ev.filter((pl.col("h") // 1000) % 10000 < rate * 10000)["entity_id"]
    p_all = samp.len() / s1.height
    q = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet")).filter(pl.col("country") == country)
                   for k in (2, 3)])
    q = q.join(truth.rename({"q_id": "entity_id"}), on="entity_id", how="left").with_columns(h=crc("entity_id"))
    q = q.filter(pl.col("s1_id").is_in(samp.implode()) |
                 (pl.col("s1_id").is_null() & (pl.col("h") % 100000 < p_all * 100000)))
    log(f"{country}: sampled S1 {samp.len()} of {s1.height} (eval half {ev.height}); records {q.height} "
        f"(unmatched {q['s1_id'].null_count()})")
    idx = CountryIndex(s1.drop("h"), "train")
    parts = [idx.chunk_pairs(q.select(Q_COLS).slice(a, CH)) for a in range(0, q.height, CH)]
    del idx
    torch.cuda.empty_cache()
    p = pl.concat(parts).join(q.select("entity_id", "s1_id"), on="entity_id", how="left")
    p = p.with_columns(label=(pl.col("entity_id_s") == pl.col("s1_id")).fill_null(False), country=pl.lit(country))
    texts = pl.concat([
        q.select("entity_id", t=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")),
        s1.filter(pl.col("entity_id").is_in(p["entity_id_s"].unique().implode())).select(
            "entity_id", t=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
    ]).with_columns(pl.col("t").str.to_lowercase().str.slice(0, 160))
    truth_s = truth.filter(pl.col("s1_id").is_in(samp.implode()))
    return p, texts, samp, truth_s


def add_folds(p):
    f = pl.when(pl.col("s1_id").is_not_null()).then(crc("s1_id") % 3).otherwise(crc("entity_id") % 3)
    return p.with_columns(fold=f.cast(pl.Int8))


def xgb_oof(df, feats, label, params, rounds):
    import xgboost as xgb
    out = np.zeros(df.height, np.float32)
    for f in range(3):
        tr, te = df.filter(pl.col("fold") != f), (df["fold"] == f).to_numpy()
        m = xgb.train(params, xgb.QuantileDMatrix(tr.select(feats).to_numpy().astype(np.float32), tr[label].cast(pl.Float32).to_numpy()),
                      rounds)
        out[te] = m.predict(xgb.DMatrix(df.filter(pl.col("fold") == f).select(feats).to_numpy().astype(np.float32)))
    return out


class CrossEncoder:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()

    def _batches(self, a, b, bs, shuffle):
        lens = np.array([len(x) + len(y) for x, y in zip(a, b)])
        if shuffle:   # similar lengths per batch (less padding), batch order shuffled
            rng = np.random.default_rng(SEED)
            order = rng.permutation(len(a))
            blocks = [order[i:i + bs * 64] for i in range(0, len(order), bs * 64)]
            bat = []
            for blk in blocks:
                blk = blk[np.argsort(lens[blk], kind="stable")]
                bat += [blk[i:i + bs] for i in range(0, len(blk), bs)]
            return [bat[j] for j in rng.permutation(len(bat))]
        order = np.argsort(lens, kind="stable")
        return [order[i:i + bs] for i in range(0, len(order), bs)]

    def fit(self, a, b, y, bs=64, lr=2e-5):
        m = self.m
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        bat = self._batches(a, b, bs, True)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(bat), pct_start=0.06)
        t0 = time.time()
        for i, ix in enumerate(bat):
            enc = self.tok([a[j] for j in ix], [b[j] for j in ix], padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = m(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(y[ix], device="cuda"))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
            if i % 500 == 0:
                log(f"    step {i}/{len(bat)} loss {loss.item():.4f} ({(i + 1) / (time.time() - t0):.1f} steps/s)")

    @torch.no_grad()
    def score(self, a, b, bs=1024):
        self.m.eval()
        out = np.empty(len(a), np.float32)
        for ix in self._batches(a, b, bs, False):
            enc = self.tok([a[j] for j in ix], [b[j] for j in ix], padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[ix] = self.m(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out


def best_rule(b, prob, truth, s1e):
    """Best of threshold and expected-F0.5 rules (same search for every method). b: ints q, s + prob column."""
    res = {}
    top = b.sort(prob, descending=True).unique("q", keep="first")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        res[f"thr {t}"] = macro_f05_df(top.filter(pl.col(prob) >= t).select("s", "q"), truth, s1e)
    for floor in (0.3, 0.5):
        for alpha in (1.0, 1.5):
            res[f"expF {floor}/{alpha}"] = macro_f05_df(decide_expf(b, prob, floor, alpha), truth, s1e)
    k = max(res, key=res.get)
    return res[k], k


def main():
    t0 = time.time()
    truth = read_truth()
    P, T, S, TR = [], [], [], []
    for c in ("US", "India"):
        p, texts, samp, truth_s = build(c, truth)
        P.append(p), T.append(texts), S.append(samp), TR.append(truth_s)
    p = add_folds(pl.concat(P))
    texts = pl.concat(T).unique("entity_id")
    samp, truth_s = pl.concat(S), pl.concat(TR)
    n_true = truth_s.height
    log(f"pairs {p.height} ({p['entity_id'].n_unique()} records), positives {int(p['label'].sum())} of {n_true} "
        f"true pairs -> shortlist recall {p['label'].sum() / n_true:.4f}  [{time.time() - t0:.0f}s]")

    xp = dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.06, max_depth=10,
              min_child_weight=20, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=SEED)
    p = p.with_columns(p_gbdt=pl.Series(xgb_oof(p, FEATURES, "label", xp, 600)))
    top = (p.sort("p_gbdt", descending=True).group_by("entity_id", maintain_order=True).head(K)
             .with_columns(g_rank=pl.int_range(pl.len()).over("entity_id").cast(pl.Float32)))
    log(f"GBDT done; top-{K} per record keeps {int(top['label'].sum())} of {n_true} true pairs "
        f"({top['label'].sum() / n_true:.4f})  [{time.time() - t0:.0f}s]")
    del p

    top = (top.join(texts.rename({"t": "tq"}), on="entity_id", how="left")
              .join(texts.rename({"entity_id": "entity_id_s", "t": "ts"}), on="entity_id_s", how="left")
              .with_columns(pl.col("tq").fill_null(""), pl.col("ts").fill_null("")))
    ce = np.zeros(top.height, np.float32)
    for f in range(3):
        tr = top.filter(pl.col("fold") != f)
        pos, neg = tr.filter("label"), tr.filter(~pl.col("label"))
        n_pos = min(pos.height, TRAIN_ROWS // 2)
        n_neg = max(0, min(neg.height, TRAIN_ROWS - n_pos))
        tr = pl.concat([pos.sample(n_pos, seed=SEED), neg.sample(n_neg, seed=SEED)])
        log(f"  fold {f}: DL training on {tr.height} pairs (positives {int(tr['label'].sum())})")
        enc = CrossEncoder()
        enc.fit(tr["tq"].to_list(), tr["ts"].to_list(), tr["label"].cast(pl.Float32).to_numpy())
        te = (top["fold"] == f).to_numpy()
        sub = top.filter(pl.col("fold") == f)
        ce[te] = enc.score(sub["tq"].to_list(), sub["ts"].to_list())
        del enc
        torch.cuda.empty_cache()
        log(f"  fold {f}: scored {sub.height} pairs  [{time.time() - t0:.0f}s]")
    top = top.with_columns(ce=pl.Series(ce), p_dl=pl.Series(1 / (1 + np.exp(-ce))))

    grp = "entity_id"
    for c in ("p_gbdt", "ce"):
        mx = pl.col(c).max().over(grp)
        second = pl.col(c).top_k(2).min().over(grp)
        other = pl.when(pl.len().over(grp) > 1).then(pl.when(pl.col(c) == mx).then(second).otherwise(mx)).otherwise(None)
        top = top.with_columns((pl.col(c) - other).alias(f"{c}_margin"),
                               pl.col(c).rank("ordinal", descending=True).over(grp).cast(pl.Float32).alias(f"{c}_rank"))
    sf = ["p_gbdt", "ce", "p_gbdt_margin", "ce_margin", "p_gbdt_rank", "ce_rank"]
    top = top.with_columns(p_stack=pl.Series(xgb_oof(top, sf, "label", dict(xp, max_depth=4, eta=0.05), 300)))

    b = top.with_columns(q=id_to_int("entity_id"), s=id_to_int("entity_id_s"))
    tr_int = truth_s.select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    results = {}
    for name, prob in (("GBDT", "p_gbdt"), ("DL (cross-encoder alone)", "p_dl"), ("STACK (GBDT + DL)", "p_stack")):
        results[name] = {}
        for c in ("all", "US", "India"):
            bb = b if c == "all" else b.filter(pl.col("country") == c)
            s1e = samp.to_frame("s1_id") if c == "all" else S[0 if c == "US" else 1].to_frame("s1_id")
            s1e = s1e.select(s=id_to_int("s1_id"))
            sc, rule = best_rule(bb, prob, tr_int.join(s1e, on="s"), s1e)
            results[name][c] = [round(sc, 5), rule]
        log(f"{name:26s} macro F0.5 all {results[name]['all'][0]:.5f} ({results[name]['all'][1]}) | "
            f"US {results[name]['US'][0]:.5f} | India {results[name]['India'][0]:.5f}")
    json.dump({"n_s1": int(samp.len()), "pairs_top": top.height, "model": MODEL, "train_rows": TRAIN_ROWS,
               "results": results}, open(os.path.join(WORK, "dlx_results.json"), "w"), indent=1)
    top.select("entity_id", "entity_id_s", "country", "fold", "label", "p_gbdt", "ce", "p_stack").write_parquet(
        os.path.join(WORK, "dlx_top.parquet"))
    log(f"DONE in {time.time() - t0:.0f}s -> dlx_results.json")


if __name__ == "__main__":
    main()
