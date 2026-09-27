"""Same-address + noise-name recall generator (shared by train validation and test application).

key(addr, country): '<house number>|<sorted canonical street words>' taken from the first address component that holds
a house number (France: 5-digit postcodes skipped). Records and S1 rows with equal (country, key) are candidates; a pair is
kept when the name differs only by generator-noise ops (ops.name_ops), and a record keeps it only if exactly one S1 qualifies.
"""
import os, re, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from rapidfuzz import fuzz
from ops import name_ops, addr_ops, LEGAL

CANON = {"avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "street": "st", "str": "st",
         "road": "rd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "terrace": "ter",
         "trail": "trl", "parkway": "pkwy", "highway": "hwy", "square": "sq", "rue": "r", "chemin": "ch", "route": "rte",
         "impasse": "imp", "allee": "all", "faubourg": "fbg", "quai": "q", "residence": "res", "north": "n", "south": "s",
         "east": "e", "west": "w", "mount": "mt", "point": "pt", "heights": "hts", "cove": "cv", "crossing": "xing",
         "saint": "st", "sainte": "ste", "floor": "fl", "building": "bldg", "suite": "ste", "apartment": "apt",
         "number": "no", "nagar": "ngr"}
DROPW = set("de des du d l la le les a en of the and et no num numero n".split())
GENERIC = set("office flat plot shop unit fl bldg ste apt room block door house h floor sector ward survey sy gala".split())
PC = re.compile(r"(?<![0-9])\d{5}(?![0-9])")
NUM = re.compile(r"(?<![0-9a-z])(\d{1,6})(?![0-9])")


def strip_acc(x):
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()


def _words(c):
    out = []
    for w in re.split(r"[^a-z0-9]+", c):
        if not w or w.isdigit() or w in DROPW:
            continue
        out.append(CANON.get(w, w))
    return out


def key(a, country):
    if not a:
        return None
    x = strip_acc(a).lower()
    if country == "France":
        x = PC.sub(" ", x)
    comps = x.split(",")
    for i, c in enumerate(comps):
        m = NUM.search(c)
        if not m:
            continue
        ws = _words(c)
        if not any(len(w) >= 3 and w not in GENERIC for w in ws) and i + 1 < len(comps):
            ws = ws + _words(comps[i + 1])
        if not any(len(w) >= 3 and w not in GENERIC for w in ws):
            return None
        ext = c + ("," + comps[i + 1] if ws is not None and len(ws) > len(_words(c)) else "")
        nums = [str(int(n)) for n in re.findall(r"\d+", ext)]
        return "-".join(nums) + "|" + " ".join(sorted(set(ws)))
    return None


def add_key(df, acol, out="k"):
    return df.with_columns(pl.Series(out, [key(a, c) for a, c in zip(df[acol].to_list(), df["country"].to_list())], dtype=pl.Utf8))


CLEAN = set("n_domain n_acronym n_squash n_word_order n_leet n_upper n_lower n_case_other n_title n_acc_add n_acc_strip "
            "n_legal_drop n_legal_dot n_legal_abbrev n_dblspace n_bracket n_hyphen n_comma_add n_comma_drop n_amp_swap n_idtag n_tradingas".split())
LEGRE = re.compile(r"\b(" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b")


def squash(x, drop_tld=False):
    x = strip_acc(x or "").lower()
    if drop_tld:
        x = re.sub(r"\.(com|net|org|fr|in|co)\b.*$", "", x)
    x = re.sub(r"(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), x)
    x = LEGRE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", x))
    return re.sub(r"[^a-z0-9]", "", x)


def judge(qn, qa, sn, sa, country):
    """-> (ok, name-op string, address-op string)."""
    o = name_ops(qn or "", sn or "")
    ok = all(x in CLEAN for x in o)
    if ok and "n_domain" in o:        # ops() does not compare the domain text with the S1 name
        a, b = squash(qn, True), squash(sn)
        ok = bool(a) and bool(b) and (a == b or fuzz.ratio(a, b) >= 85)
    if ok and "n_squash" in o:        # ops() accepts a fuzzy containment; require the squashed texts to agree
        a, b = squash(qn, True), squash(sn)
        ok = bool(a) and bool(b) and fuzz.ratio(a, b) >= 90
    ao = addr_ops(qa or "", sa or "", country)
    return ok, ";".join(sorted(o)), ";".join(sorted(ao))


def generate(rec, s1, max_block=50):
    """rec: q, qn, qa, country, k ; s1: s, sn, sa, country, k. Returns all candidate pairs with ops + a `keep` flag."""
    blk = s1.filter(pl.col("k").is_not_null()).with_columns(nb=pl.len().over("country", "k")).filter(pl.col("nb") <= max_block)
    j = rec.filter(pl.col("k").is_not_null()).join(blk, on=["country", "k"])
    # cheap prefilter: squashed names close, or record squashed text is short (acronym / initials) -> ops decide
    qs = [squash(x, True) for x in j["qn"].to_list()]
    ss = [squash(x) for x in j["sn"].to_list()]
    pre = [bool(a) and bool(b) and (len(a) <= 6 or fuzz.ratio(a, b) >= 60 or fuzz.partial_ratio(a, b) >= 80) for a, b in zip(qs, ss)]
    j = j.filter(pl.Series(pre))
    res = [judge(*t) for t in zip(j["qn"].to_list(), j["qa"].to_list(), j["sn"].to_list(), j["sa"].to_list(), j["country"].to_list())]
    j = j.with_columns(ok=pl.Series([r[0] for r in res]), nops=pl.Series([r[1] for r in res]), aops=pl.Series([r[2] for r in res]))
    j = j.with_columns(n_ok=pl.col("ok").sum().over("q"))
    return j.with_columns(keep=pl.col("ok") & (pl.col("n_ok") == 1),
                          low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
