"""Letter structure of the garble op: first letter kept? share of the S1 word's letters present in the garbled word.
Train analog (labelled, US/India) garble_in class vs the France typo set."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import words, undot_legal, LEGAL, LEGAL_CANON, _match, DESC
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fr-gathik-rest"))

pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(25)


def wdiff(qn, sn):
    qw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(qn or ""))]
    sw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(sn or ""))]
    qx = [w for w in qw if w not in LEGAL]
    sx = [w for w in sw if w not in LEGAL]
    used = [False] * len(qx)
    miss_s = []
    for x in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == x), None)
        if j is None:
            j = next((j for j, b in enumerate(qx) if not used[j] and _match(x, b)), None)
        if j is None:
            miss_s.append(x)
        else:
            used[j] = True
    return miss_s, [qx[j] for j in range(len(qx)) if not used[j]]


def rec(a, b):
    """share of a's letters (multiset) found in b"""
    ca, cb = Counter(a), Counter(b)
    return sum(min(v, cb[k]) for k, v in ca.items()) / max(len(a), 1)


def prec(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum(min(v, ca[k]) for k, v in cb.items()) / max(len(b), 1)


def feats(sn, qn):
    ms, mq = wdiff(qn, sn)
    if len(ms) != 1 or len(mq) != 1:
        return None
    a, b = ms[0], mq[0]
    return dict(a=a, b=b, first=a[:1] == b[:1], rec=rec(a, b), prec=prec(a, b), la=len(a), lb=len(b), a_desc=a in DESC)


an = pl.read_parquet(rf"{SCRATCH}/gap/train_analog.parquet")
k = pl.read_parquet(rf"{SCRATCH}/final/fr-gathik-rest/analog_cls.parquet")
an = an.join(k.select("s", "q", "k"), on=["s", "q"]).filter(pl.col("k") == "garble_in")
rows = []
for r in an.iter_rows(named=True):
    f = feats(r["sn"], r["qn"])
    if f:
        rows.append({**f, "y": r["y"], "pg": r["pg"], "src": "analog"})
fr = pl.read_parquet(rf"{SCRATCH}/final/fr-gathik-rest/fr_typo_in.parquet")
for r in fr.iter_rows(named=True):
    f = feats(r["sn"], r["qn"])
    if f:
        rows.append({**f, "y": None, "pg": r["p2"], "src": "fr_" + r["cls2"]})
    else:
        rows.append(dict(a=r["w_s"], b=r["w_q"], first=None, rec=None, prec=None, la=None, lb=None, a_desc=None, y=None, pg=r["p2"], src="fr_multi_" + r["cls2"]))
d = pl.DataFrame(rows, infer_schema_length=None)
d = d.with_columns(grp=pl.when(pl.col("src") == "analog").then(pl.format("analog_y{}", pl.col("y"))).otherwise(pl.col("src")))
print(d.group_by("grp").agg(n=pl.len(), first=pl.col("first").mean(), rec_mean=pl.col("rec").mean(), rec_lt05=(pl.col("rec") < 0.5).mean(),
                             rec_lt07=(pl.col("rec") < 0.7).mean(), prec_mean=pl.col("prec").mean(), prec_lt05=(pl.col("prec") < 0.5).mean(),
                             a_desc=pl.col("a_desc").mean(), pg=pl.col("pg").mean()).sort("grp"))
print(d.filter(pl.col("grp") == "analog_ytrue").select(pl.col("rec").cut([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]).alias("b")).group_by("b").len().sort("b"))
print(d.filter(pl.col("src").str.starts_with("fr_") & ~pl.col("src").str.contains("multi")).select(pl.col("rec").cut([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]).alias("b")).group_by("b").len().sort("b"))
d.write_parquet(rf"{SCRATCH}/final/verify/typo_letters.parquet")
