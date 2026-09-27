"""Learned name embeddings for S2/S3 rows whose name is in a non-Latin script (Indian scripts in this dataset).

Model: intfloat/multilingual-e5-small (MIT licence, 118M params), fine-tuned with in-batch
contrastive loss (MultipleNegativesRanking) on TRAIN pairs (non-Latin S2/S3 name, S1 name).
Only S1 rows in the "embedding partition" (id hash % 1000 >= 500) are used for fine-tuning, so the
other half of train stays clean for validating the downstream model.

  python embed.py train           # fine-tune -> WORK/e5_ft_addr/
  python embed.py eval            # recall@10 of the fine-tuned model on held-out rows
  python embed.py encode          # non-Latin records + S1 rows of their countries -> WORK/emb/{split}_{s1|q}.npz
  python embed.py encode_all      # every S1 row and record (second pipeline) -> WORK/emb/{split}_s1all.npz, _qall_*.npy
"""
import os
import sys
import zlib
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import WORK, log, nonlatin_countries, read_truth

BASE = "intfloat/multilingual-e5-small"
FT = os.path.join(WORK, "e5_ft_addr")
EMB = os.path.join(WORK, "emb")
MAXLEN = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def emb_partition(s1_id):
    """True for S1 rows whose matches may be used to fine-tune the embedding model."""
    return zlib.crc32(s1_id.encode()) % 1000 >= 500


def prep_text(s):
    """Model input for multilingual-e5: 'query: ' + the lowercased text."""
    return "query: " + (s or "").lower()


def text_col():
    """Embedding text: raw name + ' | ' + cleaned address (first 80 chars)."""
    return pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "),
                          pl.col("addr").fill_null("").str.slice(0, 80)]).alias("text")


class Encoder:
    """multilingual-e5 sentence encoder: mean-pooled, L2-normalised vectors."""
    def __init__(self, path):
        """Load the tokenizer and the model from `path` (hub name or saved folder), on the GPU when there is one."""
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(DEV)

    def forward(self, texts):
        """Vectors of a list of texts, with gradients (training); at most MAXLEN tokens each."""
        b = self.tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return F.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    @torch.no_grad()
    def encode(self, texts, bs=1024):
        """float16 vectors of `texts`, bs texts per batch, without gradients."""
        self.model.eval()
        out = np.empty((len(texts), self.model.config.hidden_size), np.float16)
        for a in range(0, len(texts), bs):
            with torch.autocast(DEV, dtype=torch.float16):
                out[a:a + bs] = self.forward([prep_text(t) for t in texts[a:a + bs]]).float().cpu().numpy()
        return out


def nonlatin_pairs(partition_flag):
    """Train pairs (non-Latin-script record text, S1 text) whose S1 row is in the embedding half
    (partition_flag True) or in the held-out half (False)."""
    truth = read_truth()
    q = pl.concat([pl.read_parquet(os.path.join(WORK, f"train_s{k}.parquet"),
                                   columns=["entity_id", "business_name", "addr", "name_nonlatin"]) for k in (2, 3)])
    q = q.filter("name_nonlatin").select(q_id="entity_id", q_name=text_col())
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "business_name", "addr"])
    pr = (truth.join(q, on="q_id").join(s1.select(s1_id="entity_id", s_name=text_col()), on="s1_id")
               .filter(pl.col("s1_id").map_elements(emb_partition, return_dtype=pl.Boolean) == partition_flag))
    return pr


def train(epochs=1, bs=128, lr=5e-5, max_pairs=300_000):
    """Fine-tune multilingual-e5-small on up to max_pairs (record, S1) text pairs of the embedding half: in-batch
    contrastive loss, temperature 0.05, word-embedding table frozen. Saves the model to WORK/e5_ft_addr/."""
    pr = nonlatin_pairs(True).sample(fraction=1.0, shuffle=True, seed=0).head(max_pairs)
    log(f"fine-tune pairs: {pr.height}")
    enc = Encoder(BASE)
    # freeze the 96M-param word-embedding table (4 GB GPU); the 21M transformer params are trained
    enc.model.embeddings.word_embeddings.weight.requires_grad_(False)
    enc.model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in enc.model.parameters() if p.requires_grad], lr=lr)
    steps = epochs * (pr.height // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    qn, sn = pr["q_name"].to_list(), pr["s_name"].to_list()
    enc.model.train()
    step = 0
    for ep in range(epochs):
        for a in range(0, pr.height - bs + 1, bs):
            with torch.autocast(DEV, dtype=torch.float16):
                qa = enc.forward([prep_text(t) for t in qn[a:a + bs]])
                sa = enc.forward([prep_text(t) for t in sn[a:a + bs]])
                logits = qa @ sa.T / 0.05
                labels = torch.arange(len(qa), device=DEV)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                log(f"step {step}/{steps} loss {loss.item():.4f}")
    enc.model.save_pretrained(FT)
    enc.tok.save_pretrained(FT)
    log("saved", FT)


def recall_at_k(path, n=20000, k=10):
    """Held-out (non-embedding partition) non-Latin rows: is the true S1 in the top-k of all train S1 rows of the
    countries that have non-Latin-script records (common.nonlatin_countries; India in this dataset)?"""
    pr = nonlatin_pairs(False).sample(n, seed=1)
    s1 = pl.read_parquet(os.path.join(WORK, "train_s1.parquet"), columns=["entity_id", "business_name", "addr", "country"])
    s1 = s1.filter(pl.col("country").is_in(list(nonlatin_countries("train")))).with_columns(text_col())
    enc = Encoder(path)
    S = torch.tensor(enc.encode(s1["text"].to_list()), device=DEV)
    Q = torch.tensor(enc.encode(pr["q_name"].to_list()), device=DEV)
    pos = {e: i for i, e in enumerate(s1["entity_id"].to_list())}
    tgt = torch.tensor([pos[x] for x in pr["s1_id"].to_list()], device=DEV)
    hit = 0
    for a in range(0, len(Q), 256):
        top = (Q[a:a + 256] @ S.T).topk(k, dim=1).indices
        hit += (top == tgt[a:a + 256, None]).any(1).sum().item()
    return hit / len(Q)


def encode_all():
    """Embeddings for S1 rows of every country that has non-Latin S2/S3 names, and for those S2/S3 rows."""
    os.makedirs(EMB, exist_ok=True)
    enc = Encoder(FT)
    for split in ("train", "test"):
        q = pl.concat([pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"),
                                       columns=["entity_id", "business_name", "addr", "country", "name_nonlatin"])
                       for k in (2, 3)]).filter("name_nonlatin").with_columns(text_col())
        countries = q["country"].unique().to_list()
        s1 = pl.read_parquet(os.path.join(WORK, f"{split}_s1.parquet"), columns=["entity_id", "business_name", "addr", "country"])
        s1 = s1.filter(pl.col("country").is_in(countries)).with_columns(text_col())
        for name, df in (("s1", s1), ("q", q)):
            v = enc.encode(df["text"].to_list())
            np.savez(os.path.join(EMB, f"{split}_{name}.npz"), ids=np.array(df["entity_id"].to_list()), v=v)
            log(f"{split} {name}: {v.shape}")


def encode_all_records(splits=("train", "test"), bs=2048):
    """Embeddings for EVERY S1 and S2/S3 row (audit 26 Sep: a dense top-10 search for all records, not only
    non-Latin names, lifts shortlist recall US 0.9943 -> 0.9971, India 0.9884 -> 0.9956 on top of the lexical fixes).
    S1 -> emb/{split}_s1all.npz; S2/S3 -> emb/{split}_qall_v.npy (written as a memmap) + {split}_qall_ids.npy."""
    os.makedirs(EMB, exist_ok=True)
    enc = Encoder(FT)
    for split in splits:
        s1 = pl.read_parquet(os.path.join(WORK, f"{split}_s1.parquet"), columns=["entity_id", "business_name", "addr"])
        s1 = s1.with_columns(text_col())
        np.savez(os.path.join(EMB, f"{split}_s1all.npz"), ids=np.array(s1["entity_id"].to_list()),
                 v=enc.encode(s1["text"].to_list(), bs=bs))
        log(f"{split} S1: {s1.height}")
        del s1
        q = pl.concat([pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"),
                                       columns=["entity_id", "business_name", "addr"]) for k in (2, 3)])
        np.save(os.path.join(EMB, f"{split}_qall_ids.npy"), np.array(q["entity_id"].to_list(), dtype=object))
        out = np.lib.format.open_memmap(os.path.join(EMB, f"{split}_qall_v.npy.tmp"), mode="w+", dtype=np.float16,
                                        shape=(q.height, enc.model.config.hidden_size))
        step = 500_000
        for a in range(0, q.height, step):
            out[a:a + step] = enc.encode(q[a:a + step].select(text_col())["text"].to_list(), bs=bs)
            log(f"{split} S2/S3: {min(a + step, q.height)}/{q.height}")
        out.flush()
        del out
        os.replace(os.path.join(EMB, f"{split}_qall_v.npy.tmp"), os.path.join(EMB, f"{split}_qall_v.npy"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "eval":
        log(f"recall@10 fine-tuned: {recall_at_k(FT):.4f}")
    elif cmd == "encode":        # encode_all(): non-Latin records and the S1 rows of their countries (step 2)
        encode_all()
    elif cmd == "encode_all":    # encode_all_records(): every S1 row and record (second pipeline, BER_V8=1)
        encode_all_records()
