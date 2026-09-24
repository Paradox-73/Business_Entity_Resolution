"""Text cleaning for names and addresses (polars expressions, vectorised).

Name cleaning: accents stripped, web domains turned back into words ("acme.com" -> "acme"),
fake alias prefixes dropped ("X d/b/a Acme" -> "Acme"), punctuation removed, and legal
forms / titles moved out into `name_core` so "Acme Pvt Ltd" and "Acme Private Limited" agree.

Address cleaning: local-script phrases and short forms are mapped with tables learned from
the training pairs (see learn_maps.py), junk tokens removed, numbers extracted without
leading zeros.
"""
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
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def name_exprs(col="business_name"):
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


def addr_exprs(script_map=None, abbrev_map=None, col="business_address"):
    raw = pl.col(col).fill_null("")
    x = raw.str.to_lowercase()
    if script_map:
        x = x.str.replace_many(list(script_map.keys()), list(script_map.values()))
    x = _ascii_lower(x).str.replace_all(_NONLATIN, " ")
    x = x.str.replace_all(r"[^a-z0-9]+", " ")
    x = _squash(x.str.replace_all(_JUNK, " "))
    if abbrev_map:
        x = (x.str.split(" ").list.eval(pl.element().replace(abbrev_map)).list.join(" "))
    nums = x.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(
        pl.element().filter(pl.element() != ""))
    return [
        x.alias("addr"),
        nums.alias("addr_nums"),
        (raw.str.strip_chars() == "").alias("addr_missing"),
        raw.str.contains(_NONLATIN + "{3,}").alias("addr_nonlatin"),
    ]


def normalize(df, script_map=None, abbrev_map=None):
    return df.with_columns(name_exprs() + addr_exprs(script_map, abbrev_map))
