"""Refine classes: handle/domain-like names, garble = typo of the S1's own word, strict twins; lowercase test; US analog."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from rapidfuzz import fuzz
from common import WORK, id_to_int, log
from ops import words, undot_legal, LEGAL, NOISE, DESC, STREET, STREET_ABBR, strip_acc

T = rf"{SCRATCH}/final/fr-gathik-rest"
d = pl.read_parquet(os.path.join(T, "rest_cls.parquet"))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).filter(
    pl.col("country") == "France").select(s=id_to_int("entity_id").cast(pl.Int64), sn="business_name", sa="business_address")
voc = Counter()
for n in s1["sn"].to_list():
    for w in set(words(undot_legal(n or ""))):
        voc[w] += 1
top = [w for w, c in voc.most_common(3000) if len(w) > 2 and w not in LEGAL]
STREETW = set(STREET) | set(STREET_ABBR) | {"r", "rue", "av", "avenue", "bd", "boulevard", "chemin", "ch", "route", "rte", "impasse", "imp",
                                             "allee", "all", "place", "pl", "quai", "cours", "crs", "de", "du", "des", "la", "le", "les", "l", "d",
                                             "bis", "ter", "no", "n", "st", "saint", "sainte"}


def street_key(a):
    if not a:
        return None, None
    for c in a.split(","):
        x = re.sub(r"(?<![0-9])\d{5}(?![0-9])", " ", strip_acc(c).lower())
        m = re.search(r"(?<![0-9a-z])0*(\d{1,5})", x)
        if m:
            return m.group(1), " ".join(sorted(w for w in re.split(r"[^a-z]+", x) if w and w not in STREETW and len(w) > 1))
    return None, None


def core(n):
    return " ".join(sorted(w for w in words(undot_legal(n or "")) if w not in LEGAL))


def legal(n):
    return " ".join(sorted(w for w in words(undot_legal(n or "")) if w in LEGAL))


sk = [street_key(a) for a in s1["sa"].to_list()]
s1 = s1.with_columns(hn=pl.Series([k[0] for k in sk]), st=pl.Series([k[1] for k in sk]),
                     core=pl.Series([core(n) for n in s1["sn"].to_list()]), leg=pl.Series([legal(n) for n in s1["sn"].to_list()]))
# S1-S1 twins: same street and same core words
tw = s1.filter(pl.col("st").is_not_null() & (pl.col("st") != "")).group_by("st", "core").agg(n_twin=pl.len())
d = d.join(s1.select("s", "core", "leg"), on="s", how="left")
d = d.join(tw, on=["st", "core"], how="left").with_columns((pl.col("n_twin").fill_null(1) - 1).alias("n_twin"))
# record explained exactly by another S1 on the same street
d = d.with_columns(qcore=pl.Series([core(n) for n in d["qn"].to_list()]), qleg=pl.Series([legal(n) for n in d["qn"].to_list()]))
ex = s1.filter(pl.col("st").is_not_null() & (pl.col("st") != "")).select("st", qcore="core", s_ex="s", ex_leg="leg", ex_hn="hn")
e = d.select("s", "q", "st", "qcore").join(ex, on=["st", "qcore"]).filter(pl.col("s_ex") != pl.col("s")).group_by("s", "q").agg(
    n_rec_twin=pl.len(), ex_s=pl.col("s_ex").first(), ex_leg=pl.col("ex_leg").first())
d = d.join(e, on=["s", "q"], how="left").with_columns(pl.col("n_rec_twin").fill_null(0))

# handle / domain-like record names (single token, @/# prefix, 'com' suffix): lowercased by the op itself
d = d.with_columns(handle=pl.col("qn").str.strip_chars().str.contains(r"^[@#]|^\S+$|com$"))


def garble_info(ws, wq):
    """for each record word not matched, is the S1's missing word its closest word (typo of the S1 word)?"""
    ws, wq = (ws or "").split(), (wq or "").split()
    if not wq:
        return None, None
    own, best_other = [], []
    for b in wq:
        so = max((fuzz.ratio(a, b) for a in ws), default=0)
        cands = [w for w in top if w not in ws and abs(len(w) - len(b)) <= 3]
        bo = max((fuzz.ratio(w, b) for w in cands), default=0)
        own.append(so)
        best_other.append(bo)
    return min(own), max(best_other)


gi = [garble_info(a, b) for a, b in zip(d["w_s"].to_list(), d["w_q"].to_list())]
d = d.with_columns(own_sim=pl.Series([g[0] for g in gi], dtype=pl.Float64), other_sim=pl.Series([g[1] for g in gi], dtype=pl.Float64))
d = d.with_columns(typo_own=(pl.col("own_sim") >= pl.col("other_sim")) & (pl.col("own_sim") >= 50))


def cls2(r):
    c = r["cls"]
    if r["handle"] and c in ("G_garble_in", "D_drop_words", "N_noise_in", "R_realword_in", "X_desc_in", "M_mixed"):
        return "H_handle"
    if c == "G_garble_in":
        return "G_typo_own" if r["typo_own"] else "G_garble_other"
    return c


d = d.with_columns(cls2=pl.Series([cls2(r) for r in d.iter_rows(named=True)]))
d.write_parquet(os.path.join(T, "rest_cls2.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(20)
lok = ~pl.col("handle") & pl.col("lowtest_ok")
log(str(d.group_by("cls2").agg(n=pl.len(), p08=(pl.col("p2") >= 0.8).sum(), low=pl.col("low").filter(lok).sum(), n_ok=lok.sum(),
                                   up=pl.col("numup").sum(), down=pl.col("opsF").str.contains("a_num_down").sum(),
                                   legbad=pl.col("legalbad").sum(), empty_row=(pl.col("n_v10b") == 0).sum(),
                                   s1twin=(pl.col("n_twin") > 0).sum(), rectwin=(pl.col("n_rec_twin") > 0).sum()).sort("n", descending=True)))
