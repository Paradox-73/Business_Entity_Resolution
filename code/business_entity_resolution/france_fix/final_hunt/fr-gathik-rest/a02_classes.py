"""Classify the 2,441 rest pairs by the words that change; twin check (other France S1 on the same street); vocabulary."""
import os, sys, re
from collections import Counter
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from rapidfuzz import fuzz
from common import WORK, id_to_int, log
from ops import words, undot_legal, LEGAL, NOISE, DESC, STOPW, STREET, STREET_ABBR, strip_acc

T = r"C:/ber_scratch/final/fr-gathik-rest"
rest = pl.read_parquet(os.path.join(T, "rest.parquet"))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).filter(
    pl.col("country") == "France").select(s=id_to_int("entity_id").cast(pl.Int64), sn="business_name", sa="business_address")
log(f"France S1 {s1.height}")

# vocabulary of France S1 names (non-legal words)
voc = Counter()
for n in s1["sn"].to_list():
    for w in set(words(undot_legal(n or ""))):
        voc[w] += 1
NOISEX = NOISE | {"&", "+", "et", "and", "compagnie", "cie", "fils", "freres", "associes", "service", "services", "groupe"}
STREETW = set(STREET) | set(STREET_ABBR) | {"r", "rue", "av", "avenue", "bd", "boulevard", "chemin", "ch", "route", "rte", "impasse", "imp",
                                             "allee", "all", "place", "pl", "quai", "cours", "crs", "de", "du", "des", "la", "le", "les", "l", "d",
                                             "bis", "ter", "no", "n", "st", "saint", "sainte"}


def street_key(a):
    """(house number, street words) of the first comma component holding a number; None if none."""
    if not a:
        return None, None
    for c in a.split(","):
        x = strip_acc(c).lower()
        x2 = re.sub(r"(?<![0-9])\d{5}(?![0-9])", " ", x)
        m = re.search(r"(?<![0-9a-z])0*(\d{1,5})", x2)
        if m:
            ws = sorted(w for w in re.split(r"[^a-z]+", x2) if w and w not in STREETW and len(w) > 1)
            return m.group(1), " ".join(ws)
    return None, None


sk = [street_key(a) for a in s1["sa"].to_list()]
s1 = s1.with_columns(hn=pl.Series([k[0] for k in sk]), st=pl.Series([k[1] for k in sk]))
rs = rest.select("s").join(s1.select("s", "st"), on="s", how="left")
streets = rs.filter(pl.col("st").is_not_null() & (pl.col("st") != ""))["st"].unique()
same_st = s1.filter(pl.col("st").is_in(streets.implode()))
log(f"S1 rows on the streets of rest pairs: {same_st.height}")


def core(n):
    return " ".join(w for w in words(undot_legal(n or "")) if w not in LEGAL)


def legal(n):
    return " ".join(sorted(w for w in words(undot_legal(n or "")) if w in LEGAL))


d = rest.join(s1.select("s", "hn", "st"), on="s", how="left")
by_st = {}
for r in same_st.iter_rows(named=True):
    by_st.setdefault(r["st"], []).append(r)
out = []
for r in d.iter_rows(named=True):
    qc = core(r["qn"])
    sim_p = fuzz.token_sort_ratio(qc, core(r["sn"]))
    n_same_addr, n_same_st, best_o, best_o_s, best_o_n = 0, 0, -1.0, None, None
    for o in by_st.get(r["st"], []) if r["st"] else []:
        if o["s"] == r["s"]:
            continue
        n_same_st += 1
        if o["hn"] == r["hn"]:
            n_same_addr += 1
        sm = fuzz.token_sort_ratio(qc, core(o["sn"]))
        if sm > best_o:
            best_o, best_o_s, best_o_n = sm, o["s"], o["sn"]
    # word classes of the changed words
    wq = r["w_q"].split() if r["w_q"] else []
    ws = r["w_s"].split() if r["w_s"] else []
    q_noise = all(w in NOISEX for w in wq)
    q_desc = any(w in DESC for w in wq)
    q_real = [w for w in wq if w not in NOISEX and w not in DESC and voc.get(w, 0) >= 5]
    q_garble = [w for w in wq if w not in NOISEX and w not in DESC and voc.get(w, 0) < 5]
    s_stop_only = all(w in STOPW or w in ("de", "du", "des", "la", "le", "les", "l", "d") for w in ws)
    out.append(dict(sim_p=sim_p, n_same_addr=n_same_addr, n_same_st=n_same_st, best_o=best_o, best_o_s=best_o_s, best_o_n=best_o_n,
                    q_noise=q_noise, q_desc=q_desc, q_real=" ".join(q_real), q_garble=" ".join(q_garble), s_stop_only=s_stop_only,
                    q_leg=legal(r["qn"]), s_leg=legal(r["sn"])))
d = pl.concat([d, pl.DataFrame(out)], how="horizontal")


def cls(r):
    if not r["wordchg"]:
        if r["legalbad"]:
            return "L_legal_change_add"
        if r["numup"]:
            return "U_num_up_only"
        return "P_lowp_noise_only"
    if r["w_q"] == "" or r["w_q"] is None:
        return "D_drop_stopword_only" if r["s_stop_only"] else "D_drop_words"
    if r["q_desc"]:
        return "X_desc_in"
    if r["q_noise"]:
        return "N_noise_in"
    if r["q_real"] and not r["q_garble"]:
        return "R_realword_in"
    if r["q_garble"] and not r["q_real"]:
        return "G_garble_in"
    return "M_mixed"


d = d.with_columns(cls=pl.Series([cls(r) for r in d.iter_rows(named=True)]))
d.write_parquet(os.path.join(T, "rest_cls.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(260)
lok = pl.col("lowtest_ok")
log(str(d.group_by("cls").agg(n=pl.len(), p_ge08=(pl.col("p2") >= 0.8).mean(), low_ok=pl.col("low").filter(lok).mean(), n_ok=lok.sum(),
                                  up=pl.col("numup").mean(), down=pl.col("opsF").str.contains("a_num_down").mean(),
                                  s_has=(pl.col("n_v10b") > 0).mean(), same_addr_twin=(pl.col("n_same_addr") > 0).mean(),
                                  closer_twin=(pl.col("best_o") > pl.col("sim_p")).mean(), p=pl.col("p2").mean(), pgx=pl.col("pgx").mean())
        .sort("n", descending=True)))
