import sys, re, random, collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import words, STOPW, LEGAL, undot_legal
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
D = rf"{SCRATCH}/final/same-address/"
t = pl.read_parquet(D + "test_cands.parquet", columns=["q", "qn", "k", "s", "sn", "nb", "country", "keep"]).filter(pl.col("country") == "France")
core = lambda n: [w for w in words(undot_legal(n or "")) if w not in LEGAL and w not in STOPW]
def ini(n):
    c = core(n); return "".join(w[0] for w in c) if len(c) >= 2 else None
def tok(n):
    c = core(n)
    if len(c) == 1 and 2 <= len(c[0]) <= 6 and c[0].isalpha() and re.sub(r"[^A-Za-z]", "", n).upper() == re.sub(r"[^A-Za-z]", "", n): return c[0]
    return None
t = t.with_columns(tok=pl.col("qn").map_elements(tok, return_dtype=pl.Utf8), ini=pl.col("sn").map_elements(ini, return_dtype=pl.Utf8))
blocks = t.group_by("k").agg(pl.col("ini").unique(maintain_order=True).alias("inis"), pl.col("s").n_unique().alias("nS"))
# S1 initials per key must be deduplicated per s
blocks = t.unique(["k", "s"]).group_by("k").agg(inis=pl.col("ini"), nS=pl.len())
recs = t.filter(pl.col("tok").is_not_null()).unique("q").select("q", "tok", "k")
recs = recs.join(blocks, on="k")
print("acronym-like records", recs.height)
def uniq_exact(tk, inis): return sum(1 for x in inis if x == tk) == 1
obs = sum(uniq_exact(a, b) for a, b in zip(recs["tok"], recs["inis"]))
print("observed records with exactly one exact-initial S1 at own key:", obs)
# null 1: give each record the block of another random record with the same S1 count
random.seed(0)
bysize = collections.defaultdict(list)
for kk, inis, n in zip(blocks["k"], blocks["inis"], blocks["nS"]): bysize[n].append(inis)
allb = list(zip(blocks["inis"], blocks["nS"]))
sizes = sorted(bysize)
res = []
for rep in range(30):
    c = 0
    for a, n in zip(recs["tok"], recs["nS"]):
        pool = bysize[n] if len(bysize[n]) >= 5 else [b for b, m in allb if abs(m - n) <= 2]
        c += uniq_exact(a, random.choice(pool))
    res.append(c)
print("null (random other block of same size): mean", sum(res) / len(res), "min", min(res), "max", max(res), " -> chance rate per record", sum(res) / len(res) / recs.height)
# by acronym length
recs = recs.with_columns(L=pl.col("tok").str.len_chars(), ex=pl.Series([uniq_exact(a, b) for a, b in zip(recs["tok"], recs["inis"])]))
for L in (2, 3, 4):
    sub = recs.filter(pl.col("L") == L) if L < 4 else recs.filter(pl.col("L") >= 4)
    nul = []
    for rep in range(30):
        c = 0
        for a, n in zip(sub["tok"], sub["nS"]):
            pool = bysize[n] if len(bysize[n]) >= 5 else [b for b, m in allb if abs(m - n) <= 2]
            c += uniq_exact(a, random.choice(pool))
        nul.append(c)
    print(f"len {L}{'+' if L==4 else ''}: records {sub.height}, observed unique exact {sub['ex'].sum()}, null mean {sum(nul)/len(nul):.1f}  chance rate {sum(nul)/len(nul)/sub.height:.4f}")
recs.select("q", "tok", "L", "nS", "ex").write_parquet(rf"{SCRATCH}/final/verify/sa/acr_recs.parquet")
