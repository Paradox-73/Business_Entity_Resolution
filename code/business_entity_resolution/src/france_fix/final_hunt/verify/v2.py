import sys, re, collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import DESC, words, STOPW, LEGAL, undot_legal
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
D = rf"{SCRATCH}/final/same-address/"
t = pl.read_parquet(D + "test_cands.parquet").filter(pl.col("country") == "France")
core = lambda n: [w for w in words(undot_legal(n or "")) if w not in LEGAL and w not in STOPW]
# ---- (A) visible single-word swaps at same key: how often does the new word keep the initial?
rows = []
for qn, sn, nops in zip(t["qn"], t["sn"], t["nops"]):
    if "n_acronym" in nops: continue
    a, b = core(sn), core(qn)
    if len(a) != len(b) or len(a) < 2: continue
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) != 1: continue
    i = diff[0]
    rows.append((a[i], b[i], a[i][0] == b[i][0], i, len(a)))
sw = pl.DataFrame(rows, schema=["old", "new", "same_init", "pos", "L"], orient="row")
print("visible single-word swaps at same key:", sw.height, " same initial:", round(sw["same_init"].mean(), 4))
print(sw.group_by("old").agg(n=pl.len(), same=pl.col("same_init").mean()).sort("n", descending=True).head(25))
print(sw.group_by("new").len().sort("len", descending=True).head(25))
sw.write_parquet(rf"{SCRATCH}/final/verify/sa/swaps.parquet")
# ---- (B) acronym-like unmatched records: exact vs one-letter-off initials against S1s at same key
acr = t.filter(pl.col("qn").map_elements(lambda n: len(core(n)) == 1 and 2 <= len(core(n)[0]) <= 6 and core(n)[0].isalpha()
                                          and re.sub(r"[^A-Za-z]", "", n).upper() == re.sub(r"[^A-Za-z]", "", n), return_dtype=pl.Boolean))
print("acronym-like unmatched France records at a key:", acr["q"].n_unique(), " pairs", acr.height)
res = []
for q, qn, sn, keep in zip(acr["q"], acr["qn"], acr["sn"], acr["keep"]):
    tok = core(qn)[0]
    c = core(sn)
    ini = "".join(w[0] for w in c)
    if len(c) < 2: 
        res.append((q, "none", None, keep)); continue
    if ini == tok:
        res.append((q, "exact", None, keep)); continue
    if len(ini) == len(tok):
        d = [i for i in range(len(tok)) if ini[i] != tok[i]]
        if len(d) == 1:
            res.append((q, "oneoff", c[d[0]], keep)); continue
    res.append((q, "none", None, keep))
r = pl.DataFrame(res, schema=["q", "kind", "word", "keep"], orient="row")
per = r.group_by("q").agg(exact=(pl.col("kind") == "exact").any(), oneoff=(pl.col("kind") == "oneoff").any(), keep=pl.col("keep").any())
print(per.group_by("exact", "oneoff", "keep").len().sort("len", descending=True))
oo = r.filter(pl.col("kind") == "oneoff").join(per.filter(~pl.col("exact")).select("q"), on="q")
print("one-off (no exact S1) records:", oo["q"].n_unique(), "; S1 word at the differing position:")
print(oo.group_by("word").len().sort("len", descending=True).head(30))
r.write_parquet(rf"{SCRATCH}/final/verify/sa/acr_kinds.parquet")
