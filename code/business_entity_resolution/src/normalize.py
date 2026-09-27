"""Text cleaning for names and addresses (polars expressions, vectorised).

Name cleaning: accents stripped, web domains turned back into words ("acme.com" -> "acme"),
fake alias prefixes dropped ("X d/b/a Acme" -> "Acme"), punctuation removed, and legal
forms / titles moved out into `name_core` so "Acme Pvt Ltd" and "Acme Private Limited" agree.

Address cleaning: local-script phrases and short forms are mapped with tables learned from
the training pairs (see learn_maps.py), junk tokens removed, numbers extracted without
leading zeros.
"""
import functools
import os
import polars as pl

LEGAL = ["private", "pvt", "limited", "ltd", "llp", "llc", "inc", "incorporated", "lnc", "corp",
         "corporation", "co", "company", "plc", "pllc", "pc", "pa", "lp", "public", "esq",
         "sarl", "sas", "sasu", "sa", "sci", "eurl", "gmbh", "l l c", "l l p", "p c", "p a"]
TITLES = ["mr", "mrs", "ms", "dr", "sri", "shri", "smt", "m s", "the"]
JUNK_ADDR = ["null", "n a", "na", "none", "nan"]

_WORDS = r"\b(?:" + "|".join(sorted(LEGAL + TITLES, key=len, reverse=True)) + r")\b"
_JUNK = r"\b(?:" + "|".join(JUNK_ADDR) + r")\b"
_DOMAIN = r"^[#@\s]*([a-z0-9\-]+)\.(?:co\.in|com|in|net|org|co|fr|biz|info)$"
_ALIAS = r"^.*?\b(?:d/b/a|a/k/a|t/a|dba|aka)\b\s*"
_NONLATIN = r"[^\x00-\x7F]"


def _ascii_lower(e):
    """Unicode NFKD, drop combining marks (accents), lowercase."""
    return e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def _squash(e):
    """Collapse runs of whitespace to one space and trim."""
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def name_exprs(col="business_name"):
    """Polars expressions of the cleaned name columns: name_full, name_core (legal forms and titles removed),
    name_ns (core without spaces) and the flags name_is_domain, name_has_alias, name_nonlatin."""
    raw = pl.col(col).fill_null("")
    low = _ascii_lower(pl.col(col)).str.strip_chars()
    is_domain = low.str.contains(_DOMAIN)
    has_alias = low.str.contains(_ALIAS)
    x = low.str.replace(_DOMAIN, "$1").str.replace(_ALIAS, "")
    x = x.str.replace_all("&", " and ").str.replace_all(r"[^\p{L}\p{N}]+", " ")
    full = _squash(x)
    core = _squash(full.str.replace_all(_WORDS, " "))
    core = pl.when(core == "").then(full).otherwise(core)
    return [
        full.alias("name_full"),
        core.alias("name_core"),
        core.str.replace_all(" ", "").alias("name_ns"),
        is_domain.alias("name_is_domain"),
        has_alias.alias("name_has_alias"),
        raw.str.contains(_NONLATIN + "{3,}").alias("name_nonlatin"),
    ]


# BER_V8=1 (audit 26 Sep on India, EXPERIMENTS.md): state codes the pooled learned map gets wrong ('tn' -> 'tennessee';
# 'dl' / 'od' unmapped while India S1 writes 'Tamil Nadu' / 'Delhi' / 'Orissa'), and ordinals ('eleventh street' in
# records vs '11th street' in S1 -> both '11 street'). STATE_CODES applies only to the rows of state_code_countries(),
# the countries whose S1 rows write these state names rather than the codes: 'TN' is Tennessee in US addresses.
V8 = os.environ.get("BER_V8") == "1"
STATE_CODES = {"tn": "tamil nadu", "dl": "delhi", "od": "orissa", "odisha": "orissa"}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
            "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
            "twentieth"]


def _words_re(words):
    """Regex matching any of `words` as a whole word, longest first."""
    return r"\b(?:" + "|".join(sorted(words, key=len, reverse=True)) + r")\b"


@functools.lru_cache(maxsize=None)
def state_code_countries():
    """Countries whose raw Source 1 addresses (train and test files, lowercased) name a state of STATE_CODES in more
    rows than they carry one of its codes. Read from the data; on 27 Sep 2026: India 382,283 rows against 656, US 190
    against 89,794, France 0 against 0, so India alone."""
    from common import raw_path
    d = pl.concat([pl.read_csv(raw_path(split, 1), separator="\t", quote_char=None, infer_schema_length=0,
                               columns=["business_address", "country"]) for split in ("train", "test")])
    a = d["business_address"].fill_null("").str.to_lowercase()
    n = (d.with_columns(full=a.str.contains(_words_re(set(STATE_CODES.values()))), code=a.str.contains(_words_re(STATE_CODES)))
          .drop_nulls("country").group_by("country").agg(pl.col("full").sum(), pl.col("code").sum()))
    return tuple(sorted(n.filter(pl.col("full") > pl.col("code"))["country"].to_list()))


def _ordinals(e):
    """'eleventh' -> '11', '11th' / '2nd' -> '11' / '2' (both sides, so the street tokens agree)."""
    e = (" " + e + " ").str.replace_many([f" {w} " for w in ORDINALS], [f" {i + 1} " for i in range(len(ORDINALS))])
    return _squash(e.str.replace_all(r"\b(\d+)(?:st|nd|rd|th)\b", "$1"))


def addr_exprs(script_map=None, abbrev_map=None, col="business_address"):
    """Polars expressions of the cleaned address columns: addr (local-script phrases mapped, accents and junk
    tokens removed, short forms expanded with the learned map), addr_nums (numbers without leading zeros) and the flags
    addr_missing, addr_nonlatin. With BER_V8=1 also STATE_CODES (rows of state_code_countries()) and the ordinals."""
    raw = pl.col(col).fill_null("")
    x = raw.str.to_lowercase()
    if script_map:
        x = x.str.replace_many(list(script_map.keys()), list(script_map.values()))
    x = _ascii_lower(x).str.replace_all(_NONLATIN, " ")
    x = x.str.replace_all(r"[^a-z0-9]+", " ")
    x = _squash(x.str.replace_all(_JUNK, " "))
    if abbrev_map and V8:
        tok = x.str.split(" ")
        x = pl.when(pl.col("country").is_in(list(state_code_countries()))).then(
            tok.list.eval(pl.element().replace(dict(abbrev_map, **STATE_CODES))).list.join(" ")).otherwise(
            tok.list.eval(pl.element().replace(abbrev_map)).list.join(" "))
    elif abbrev_map:
        x = (x.str.split(" ").list.eval(pl.element().replace(abbrev_map)).list.join(" "))
    if V8:
        x = _ordinals(x)
    nums = x.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(
        pl.element().filter(pl.element() != ""))
    return [
        x.alias("addr"),
        nums.alias("addr_nums"),
        (raw.str.strip_chars() == "").alias("addr_missing"),
        raw.str.contains(_NONLATIN + "{3,}").alias("addr_nonlatin"),
    ]


# ---- French rules for the countries without training labels (France in this test set): rules from the word
# differences between France records and the S1 row they confidently match on test (work log 25 Sep, EXPERIMENTS.md
# section France). The learned abbreviation map is NOT applied (it turns "de" into "delaware", "la" into "louisiana", "st"
# into "street").
FR_LEGAL = ["snc", "ei", "eirl", "scp", "selarl", "selas", "gie", "earl", "sca", "cie", "compagnie", "societe",
            "ste", "ets", "etablissements", "groupe", "fils", "freres", "and", "et", "france",
            "de", "du", "des", "la", "le", "les", "l", "d"]
FR_NAME_MAP = {"5arl": "sarl", "5as": "sas", "frs": "freres", "et": "and"}
FR_ADDR_MAP = {"r": "rue", "av": "avenue", "ave": "avenue", "st": "saint", "ste": "sainte", "all": "allee",
               "bd": "boulevard", "blvd": "boulevard", "bld": "boulevard", "imp": "impasse", "rte": "route",
               "crs": "cours", "q": "quai", "pl": "place", "ch": "chemin", "chem": "chemin", "res": "residence",
               "psg": "passage", "pass": "passage", "apt": "appartement", "app": "appartement",
               "appt": "appartement", "fbg": "faubourg", "sq": "square", "gal": "general", "mal": "marechal"}
# regions (S1) and departements (S2/S3) name the same area in different words: drop both, the city stays
FR_AREA = r"\b(?:hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france)\b"
FR_ADDR_DROP = ["no", "n", "de", "du", "des", "la", "le", "les", "l", "d"]


def _fr_acronyms(e):
    """'s.a.r.l.' -> 'sarl', 'e.u.r.l' -> 'eurl' (a dot after a single letter is removed)."""
    return e.str.replace_all(r"\b([a-z])\.", "$1")


def fr_name_exprs(col="business_name"):
    """As name_exprs, with French rules: dotted legal forms joined; French legal forms and articles also
    removed from name_core. Used only by 'prep.py unlabelled' (v6; not in the final file)."""
    raw = pl.col(col).fill_null("")
    low = _fr_acronyms(_ascii_lower(pl.col(col)).str.strip_chars())
    is_domain = low.str.contains(_DOMAIN)
    has_alias = low.str.contains(_ALIAS)
    x = low.str.replace(_DOMAIN, "$1").str.replace(_ALIAS, "")
    x = x.str.replace_all("&", " and ").str.replace_all(r"[^\p{L}\p{N}]+", " ")
    x = _squash(x).str.split(" ").list.eval(pl.element().replace(FR_NAME_MAP)).list.join(" ")
    full = _squash(x)
    words = r"\b(?:" + "|".join(sorted(LEGAL + TITLES + FR_LEGAL, key=len, reverse=True)) + r")\b"
    core = _squash(full.str.replace_all(words, " "))
    core = pl.when(core == "").then(full).otherwise(core)
    return [
        full.alias("name_full"),
        core.alias("name_core"),
        core.str.replace_all(" ", "").alias("name_ns"),
        is_domain.alias("name_is_domain"),
        has_alias.alias("name_has_alias"),
        raw.str.contains(_NONLATIN + "{3,}").alias("name_nonlatin"),
    ]


def fr_addr_exprs(col="business_address"):
    """As addr_exprs, with French rules: street abbreviations expanded, region / departement names and
    articles dropped, no learned map. Used only by 'prep.py unlabelled' (v6; not in the final file)."""
    raw = pl.col(col).fill_null("")
    x = _ascii_lower(pl.col(col)).str.replace_all(_NONLATIN, " ")
    x = _squash(x.str.replace_all(r"[^a-z0-9]+", " ").str.replace_all(_JUNK, " "))
    x = _squash(x.str.replace_all(FR_AREA, " "))
    x = (x.str.split(" ").list.eval(pl.element().replace(FR_ADDR_MAP))
          .list.eval(pl.element().filter(~pl.element().is_in(FR_ADDR_DROP))).list.join(" "))
    nums = x.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(
        pl.element().filter(pl.element() != ""))
    return [
        x.alias("addr"),
        nums.alias("addr_nums"),
        (raw.str.strip_chars() == "").alias("addr_missing"),
        raw.str.contains(_NONLATIN + "{3,}").alias("addr_nonlatin"),
    ]


def normalize(df, script_map=None, abbrev_map=None, french=False):
    """df with the cleaned name and address columns added (French rules when french=True)."""
    if french:
        return df.with_columns(fr_name_exprs() + fr_addr_exprs())
    return df.with_columns(name_exprs() + addr_exprs(script_map, abbrev_map))
