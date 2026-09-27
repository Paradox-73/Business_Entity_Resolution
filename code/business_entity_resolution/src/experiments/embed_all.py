"""Retriever for ALL records (GPU job; also runs on the 4 GB laptop GPU with the small model).

Fine-tunes a multilingual e5 model on every training match (name + address text), with in-batch negatives
plus one mined hard negative per pair (the wrong S1 our stage-1 model ranked first), then encodes every
S1/S2/S3 record of train and test and writes, for every S2/S3 record, its top-K S1 rows of the same country.

Only S1 rows with crc32(id) % 1000 >= 500 contribute training pairs (same leakage rule as embed.py), so
the other half stays clean for validation.

  python embed_all.py train  [model]   # default intfloat/multilingual-e5-base (MIT, 278M); -small on 4 GB
  python embed_all.py encode
  python embed_all.py search [K]       # -> WORK/emb_all/{split}_topk.parquet  (q, s, emb2_cos)
  python embed_all.py eval             # recall@10 on held-out rows, all countries
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import zlib
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import WORK, log, read_truth, id_to_int

OUT = os.path.join(WORK, "emb_all")
FT = os.path.join(OUT, "model")
MAXLEN = 64
DEV = "cuda"
LIMIT = int(os.environ.get("BER_EMB_LIMIT", "0"))   # smoke test: >0 = only this many pairs / rows per file


def text(df):
    """'query: <lowercased raw name> | <cleaned address, first 80 chars>' (same format as embed.py)."""
    return ("query: " + df["business_name"].fill_null("").str.to_lowercase() + " | " +
            df["addr"].fill_null("").str.slice(0, 80)).to_list()


def load(split, k):
    return pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"),
                           columns=["entity_id", "business_name", "addr", "country"])


class Enc:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).to(DEV)

    def fwd(self, t):
        b = self.tok(t, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = self.m(**b).last_hidden_state
        a = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return F.normalize((h * a).sum(1) / a.sum(1), dim=-1)

    @torch.no_grad()
    def encode(self, t, bs=1024):
        self.m.eval()
        out = np.empty((len(t), self.m.config.hidden_size), np.float16)
        for i in range(0, len(t), bs):
            with torch.autocast("cuda", dtype=torch.float16):
                out[i:i + bs] = self.fwd(t[i:i + bs]).float().cpu().numpy()
            if i % (bs * 500) == 0:
                log(f"  encoded {i}/{len(t)}")
        return out


def train(base="intfloat/multilingual-e5-base", max_pairs=2_000_000, bs=256, lr=3e-5):
    max_pairs = LIMIT or int(os.environ.get("BER_EMB_PAIRS", max_pairs))
    tr = read_truth().filter(pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 >= 500,
                                                          return_dtype=pl.Boolean))
    q = pl.concat([load("train", k) for k in (2, 3)])
    q = q.with_columns(t=pl.Series(text(q)))
    s1 = load("train", 1)
    s1 = s1.with_columns(t=pl.Series(text(s1)))
    pr = (tr.join(q.select(q_id="entity_id", qt="t"), on="q_id").join(s1.select(s1_id="entity_id", st="t"), on="s1_id")
            .sample(fraction=1.0, shuffle=True, seed=0).head(max_pairs))
    # hard negative: the S1 our stage-1 model ranked first for this record when it was wrong (if any)
    oof = os.path.join(WORK, "models", "full", "stage1_base.parquet")
    if os.path.exists(oof):
        hn = (pl.read_parquet(oof, columns=["q", "s", "p1", "label"]).filter(~pl.col("label"))
                .sort("p1", descending=True).unique("q", keep="first").select("q", hs="s"))
        s1i = s1.select(hs=id_to_int("entity_id"), ht="t")
        pr = (pr.with_columns(q=id_to_int("q_id")).join(hn, on="q", how="left").join(s1i, on="hs", how="left"))
    else:
        pr = pr.with_columns(ht=pl.lit(None, pl.Utf8))
    log(f"pairs {pr.height}, with hard negative {pr['ht'].is_not_null().sum()}")
    enc = Enc(base)
    if torch.cuda.get_device_properties(0).total_memory < 8e9:     # 4 GB laptop: freeze word table, checkpoint
        enc.m.embeddings.word_embeddings.weight.requires_grad_(False)
        enc.m.gradient_checkpointing_enable()
        bs = min(bs, 96)
    opt = torch.optim.AdamW([p for p in enc.m.parameters() if p.requires_grad], lr=lr)
    steps = pr.height // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    qt, st, ht = pr["qt"].to_list(), pr["st"].to_list(), pr["ht"].to_list()
    enc.m.train()
    for i in range(steps):
        a, b = i * bs, (i + 1) * bs
        negs = [x for x in ht[a:b] if x]
        with torch.autocast("cuda", dtype=torch.float16):
            qa, sa = enc.fwd(qt[a:b]), enc.fwd(st[a:b] + negs)
            logits = qa @ sa.T / 0.05
            loss = F.cross_entropy(logits, torch.arange(len(qa), device=DEV))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
        if i % 200 == 0:
            log(f"step {i}/{steps} loss {loss.item():.4f}")
    os.makedirs(FT, exist_ok=True)
    enc.m.save_pretrained(FT)
    enc.tok.save_pretrained(FT)


def encode():
    enc = Enc(FT)
    for split in ("train", "test"):
        for k in (1, 2, 3):
            df = load(split, k)
            if LIMIT:
                df = df.head(LIMIT)
            v = enc.encode(text(df))
            np.save(os.path.join(OUT, f"{split}_s{k}.npy"), v)
            df.select(q=id_to_int("entity_id"), country="country").write_parquet(os.path.join(OUT, f"{split}_s{k}_ids.parquet"))
            log(f"{split} s{k}: {v.shape}")


def search(k=20):
    for split in ("train", "test"):
        ids1 = pl.read_parquet(os.path.join(OUT, f"{split}_s1_ids.parquet")).with_row_index("i")
        V1 = np.load(os.path.join(OUT, f"{split}_s1.npy"))
        parts = []
        for src in (2, 3):
            idq = pl.read_parquet(os.path.join(OUT, f"{split}_s{src}_ids.parquet")).with_row_index("i")
            VQ = np.load(os.path.join(OUT, f"{split}_s{src}.npy"))
            for c in idq["country"].unique().to_list():
                si = ids1.filter(pl.col("country") == c)
                if si.height == 0:
                    continue
                S = torch.tensor(V1[si["i"].to_numpy()], device=DEV)
                qi = idq.filter(pl.col("country") == c)
                Qv = VQ[qi["i"].to_numpy()]
                sid = si["q"].to_numpy()
                qid = qi["q"].to_numpy()
                qb = 2048 if torch.cuda.get_device_properties(0).total_memory > 8e9 else 256   # 4 GB laptop
                for a in range(0, len(Qv), qb):
                    sc, ix = (torch.tensor(Qv[a:a + qb], device=DEV) @ S.T).topk(k, dim=1)
                    parts.append(pl.DataFrame({"q": np.repeat(qid[a:a + qb], k), "s": sid[ix.cpu().numpy().ravel()],
                                               "emb2_cos": sc.float().cpu().numpy().ravel()}))
                log(f"{split} S{src} {c}: {len(Qv)} records searched")
        pl.concat(parts).write_parquet(os.path.join(OUT, f"{split}_topk.parquet"))


def evaluate():
    t = pl.read_parquet(os.path.join(OUT, "train_topk.parquet"))
    tr = read_truth().filter(pl.col("s1_id").map_elements(lambda x: zlib.crc32(x.encode()) % 1000 < 500,
                                                          return_dtype=pl.Boolean)).select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
    for kk in (10, 20):
        top = t.sort("emb2_cos", descending=True).group_by("q", maintain_order=True).head(kk)
        log(f"held-out recall@{kk}: {tr.join(top, on=['q', 's']).height / tr.height:.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "train":
        train(*(sys.argv[2:3]))
    elif cmd == "encode":
        encode()
    elif cmd == "search":
        search(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "eval":
        evaluate()
