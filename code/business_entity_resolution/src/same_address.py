"""Same-address rule: a third source of candidate pairs (and matches) for the countries without training labels
(common.unlabelled_countries; France in this test set). It regenerates sets/fr_same_address_safe.parquet.

Why: the searches of candidates.py find a pair when the name or address text is similar. A record whose name the data
generator rewrote as an acronym ('AC' for 'Animation Culture SCI') or a web domain shares almost no name text with its
S1 row, and at a busy address its S1 row can fall outside the searches' top-k lists. This rule instead compares each
record that the base submission leaves unmatched with the S1 rows at exactly the same address, and accepts the pair
when the two names differ only by the kinds of noise the generator puts on true copies.

The rule, for every Source 2/3 record of a country without training labels that the base file leaves unmatched:
  1. Address key (address_key): the first comma-separated address component that holds a number gives the house
     number(s) and the set of street words (accents stripped, lowercase, street types shortened: avenue -> ave,
     rue -> r, ...; filler words de/du/la/of/the/... dropped). A standalone 5-digit number is read as a postcode and
     ignored. When the component has no word of 3+ letters other than unit words (office, flat, bldg, ...), the words
     and numbers of the next component are added.
  2. Same address: the S1 rows of the same country with the same key. Keys shared by more than MAX_BLOCK (50) S1 rows
     are skipped.
  3. Cheap prefilter: both squashed names (squash) non-empty, and the record's squashed name has at most 6 characters
     (acronym, initials) or rapidfuzz ratio >= 60 or partial_ratio >= 80 with the S1 one.
  4. Name noise only (name_judge): the change detector name_ops (france_fix/artifacts/census/ops.py, the detector
     france_recall.py also uses) finds only changes of CLEAN_NAME_OPS between the record name and the S1 name: acronym
     or initials, web domain, squashed words, word order, digits for letters, case, accents, brackets, hyphens, commas,
     double spaces, '&' <-> 'and', an id tag, 'trading as', and the legal form dropped, dotted or abbreviated. A
     web-domain name must squash to the S1 name (equal or ratio >= 85), a squashed name must agree with it (ratio >= 90).
  5. Unique: exactly one S1 row at the address passes step 4 for the record.
  6. Not re-added: the pair is in none of the `exclude` pair sets (the France vetoes applied earlier in the build), and
     the second pipeline does not match the record to a different S1 row.
  7. Safety filter (safe):
     - an acronym name needs an address with exactly one S1 row;
     - any other name needs: a web-domain name that squashes to exactly the S1 name; no "street twin" (another S1 row
       of the country with the same squashed name on the same street words, i.e. a branch at another number); and no
       changed place, dropped place or changed street component in the address (ops.addr_ops, component level).
  8. The S1 row already has at least one match in the base file: the rule never makes an empty S1 row non-empty.

Evidence (27 Sep 2026, the analysis scripts in france_fix/final_hunt/same-address/ and verify/v1.py-v8.py):
  - labelled analog: on the held-out half of train, for the records our held-out prediction leaves unmatched, steps 1-5
    gave 651 pairs (US 509, India 142), 99.85% of them true (same-address/train_eval.py). That run kept 5-digit
    numbers in the US/India keys, where they are house numbers ('19438 Best Road');
  - on test with base v10b: 577,492 unmatched France records, 538,129 of them with an address key; steps 2-3 gave
    42,511 pairs, step 4 849, step 5 709, step 6 693, step 7 309 and step 8 298 (239 acronym names, 59 others).
    This module reproduces those 298 pairs; build_final.py stops if they differ from sets/fr_same_address_safe.parquet.
  - Address ops (aops) are computed without a country, so the house-number ops (a_num_*) of 34 of the 42,511 test
    candidate pairs differ from the analysis tables, which read a 5-digit postcode there; the rule uses only the
    component ops, which are identical on all 42,511.

  python same_address.py <base matching_results.tsv> <second pipeline matching_results.tsv> <out.parquet>
                         [--exclude veto.parquet ...] [--ref reference.parquet]
e.g. python same_address.py $B/v10b/matching_results.tsv $W/gathik/v8_matching_results.tsv $B/same_address_pairs.parquet
         --exclude ../sets/desc_veto_set.parquet ../sets/fp_veto_set.parquet ../sets/moreveto_stem2_set.parquet
         --ref ../sets/fr_same_address_safe.parquet
"""
import argparse
import os
import re
import sys
import time
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from apply_pair_sets import pairs
from common import WORK, id_to_int, is_unlabelled, log

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "france_fix", "artifacts", "census"))
from ops import LEGAL, addr_ops, name_ops  # noqa: E402  (the change detector; called without its country argument)

MAX_BLOCK = 50          # addresses shared by more S1 rows are skipped (step 2)
PRE_SHORT, PRE_RATIO, PRE_PARTIAL = 6, 60, 80        # prefilter (step 3)
DOMAIN_RATIO, SQUASH_RATIO = 85, 90                   # name checks (step 4)

# ---- address key (step 1)
STREET_CANON = {"avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "street": "st",
                "str": "st", "road": "rd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl",
                "terrace": "ter", "trail": "trl", "parkway": "pkwy", "highway": "hwy", "square": "sq", "rue": "r",
                "chemin": "ch", "route": "rte", "impasse": "imp", "allee": "all", "faubourg": "fbg", "quai": "q",
                "residence": "res", "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt",
                "point": "pt", "heights": "hts", "cove": "cv", "crossing": "xing", "saint": "st", "sainte": "ste",
                "floor": "fl", "building": "bldg", "suite": "ste", "apartment": "apt", "number": "no",
                "nagar": "ngr"}
DROP_WORDS = set("de des du d l la le les a en of the and et no num numero n".split())
UNIT_WORDS = set("office flat plot shop unit fl bldg ste apt room block door house h floor sector ward survey sy gala".split())
POSTCODE = re.compile(r"(?<![0-9])\d{5}(?![0-9])")
HOUSE_NUMBER = re.compile(r"(?<![0-9a-z])(\d{1,6})(?![0-9])")

# ---- names (step 4)
CLEAN_NAME_OPS = set("n_domain n_acronym n_squash n_word_order n_leet n_upper n_lower n_case_other n_title n_acc_add "
                     "n_acc_strip n_legal_drop n_legal_dot n_legal_abbrev n_dblspace n_bracket n_hyphen n_comma_add "
                     "n_comma_drop n_amp_swap n_idtag n_tradingas".split())
LEGAL_RE = re.compile(r"\b(" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b")

# ---- safety filter (step 7): address component changes that reject a non-acronym pair
PLACE_STREET_CHANGE = r"a_chg:place|a_drop:place|a_chg:street"


def strip_acc(x):
    """Text without accents (NFKD, non-ASCII dropped)."""
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()


def _street_words(component):
    """Words of one address component: no pure numbers, no filler words, street types shortened."""
    out = []
    for w in re.split(r"[^a-z0-9]+", component):
        if not w or w.isdigit() or w in DROP_WORDS:
            continue
        out.append(STREET_CANON.get(w, w))
    return out


def _has_street_word(ws):
    """True if a word of 3+ letters other than a unit word is present."""
    return any(len(w) >= 3 and w not in UNIT_WORDS for w in ws)


def address_key(addr):
    """'<numbers joined by ->|<sorted distinct street words>' of the first address component holding a number, or None
    (no address, no number, or no street word of 3+ letters other than unit words). Example:
    '4 Rue du Traité de Lisbonne, Pornic, Pays de la Loire' -> '4|lisbonne r traite'."""
    if not addr:
        return None
    comps = POSTCODE.sub(" ", strip_acc(addr).lower()).split(",")
    for i, c in enumerate(comps):
        if not HOUSE_NUMBER.search(c):
            continue
        ws = _street_words(c)
        if not _has_street_word(ws) and i + 1 < len(comps):
            ws = ws + _street_words(comps[i + 1])
        if not _has_street_word(ws):
            return None
        text = c + ("," + comps[i + 1] if len(ws) > len(_street_words(c)) else "")
        return "-".join(str(int(n)) for n in re.findall(r"\d+", text)) + "|" + " ".join(sorted(set(ws)))
    return None


def with_key(df, addr_col):
    """df with a string column k = address_key of addr_col."""
    return df.with_columns(pl.Series("k", [address_key(a) for a in df[addr_col].to_list()], dtype=pl.Utf8))


def squash(name, drop_tld=False):
    """Name as one lowercase run of letters and digits, legal forms removed; drop_tld also cuts a web-domain ending
    ('.com', '.fr', ...) and what follows. 'Animation Culture SCI' -> 'animationculture'."""
    x = strip_acc(name or "").lower()
    if drop_tld:
        x = re.sub(r"\.(com|net|org|fr|in|co)\b.*$", "", x)
    x = re.sub(r"(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), x)      # s.a.r.l. -> sarl
    x = LEGAL_RE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", x))
    return re.sub(r"[^a-z0-9]", "", x)


def prefilter(qn, sn):
    """Step 3: cheap check before the change detector."""
    a, b = squash(qn, True), squash(sn)
    return bool(a) and bool(b) and (len(a) <= PRE_SHORT or fuzz.ratio(a, b) >= PRE_RATIO or fuzz.partial_ratio(a, b) >= PRE_PARTIAL)


def name_judge(qn, sn):
    """Step 4: (ok, ';'-joined sorted name ops). ok = every op is noise, with the extra checks for web-domain and
    squashed names (name_ops does not compare the domain text with the S1 name, and accepts a fuzzy containment for
    squashed names)."""
    o = name_ops(qn or "", sn or "")
    ok = all(x in CLEAN_NAME_OPS for x in o)
    if ok and "n_domain" in o:
        a, b = squash(qn, True), squash(sn)
        ok = bool(a) and bool(b) and (a == b or fuzz.ratio(a, b) >= DOMAIN_RATIO)
    if ok and "n_squash" in o:
        a, b = squash(qn, True), squash(sn)
        ok = bool(a) and bool(b) and fuzz.ratio(a, b) >= SQUASH_RATIO
    return ok, ";".join(sorted(o))


def unique_noise_pairs(rec, s1):
    """Steps 2-5. rec: q, qn, qa, country, k; s1: s, sn, sa, country, k. Returns every prefiltered same-address pair
    with its name ops (nops), address ops (aops), the number of S1 rows at the address (nb), ok and keep (steps 4, 5)."""
    blk = s1.filter(pl.col("k").is_not_null()).with_columns(nb=pl.len().over("country", "k")).filter(pl.col("nb") <= MAX_BLOCK)
    j = rec.filter(pl.col("k").is_not_null()).join(blk, on=["country", "k"])
    j = j.filter(pl.Series([prefilter(a, b) for a, b in zip(j["qn"].to_list(), j["sn"].to_list())], dtype=pl.Boolean))
    res = [name_judge(a, b) for a, b in zip(j["qn"].to_list(), j["sn"].to_list())]
    aops = [";".join(sorted(addr_ops(a or "", b or ""))) for a, b in zip(j["qa"].to_list(), j["sa"].to_list())]
    j = j.with_columns(ok=pl.Series([r[0] for r in res], dtype=pl.Boolean), nops=pl.Series([r[1] for r in res], dtype=pl.Utf8),
                       aops=pl.Series(aops, dtype=pl.Utf8))
    j = j.with_columns(n_ok=pl.col("ok").sum().over("q"))
    return j.with_columns(keep=pl.col("ok") & (pl.col("n_ok") == 1))


def street_twins(p, s1):
    """p with street_twins: the number of other S1 rows of the pair's country whose squashed name equals the pair's
    squashed S1 name and whose address key has the same street words (any house number)."""
    street = pl.col("k").str.split("|").list.get(1, null_on_oob=True)
    p = p.with_columns(sq=pl.Series([squash(x) for x in p["sn"].to_list()], dtype=pl.Utf8), st=street)
    names = p["sq"].unique().implode()
    s = s1.with_columns(sq=pl.Series([squash(x) for x in s1["sn"].to_list()], dtype=pl.Utf8), st=street).filter(pl.col("sq").is_in(names))
    tw = (p.select("s", "country", "sq", "st").unique().join(s.select(s2="s", country="country", sq="sq", st="st"), on=["country", "sq", "st"])
           .filter(pl.col("s2") != pl.col("s")).group_by("s").len("street_twins"))
    return p.join(tw, on="s", how="left").with_columns(pl.col("street_twins").fill_null(0)).drop("sq", "st")


def safe(p):
    """Step 7 on the pairs p (columns nops, aops, nb, street_twins, qn, sn)."""
    p = p.with_columns(acr=pl.col("nops").str.contains("n_acronym", literal=True),
                       dom_exact=pl.Series([squash(a, True) == squash(b) for a, b in zip(p["qn"].to_list(), p["sn"].to_list())], dtype=pl.Boolean))
    dom = pl.col("nops").str.contains("n_domain", literal=True)
    return p.filter((pl.col("acr") & (pl.col("nb") == 1)) |
                    (~pl.col("acr") & (~dom | pl.col("dom_exact")) & (pl.col("street_twins") == 0) & ~pl.col("aops").str.contains(PLACE_STREET_CHANGE)))


def build(base_tsv, second_tsv=None, exclude=()):
    """The same-address pairs for the base submission base_tsv (steps 1-8). second_tsv: the second pipeline's
    matching_results.tsv (step 6; None skips that check). exclude: pair-set parquet files (columns s, q) whose pairs
    are never added. Returns a table with s, q and the evidence columns, sorted by (s, q)."""
    t0 = time.time()
    base = pairs(base_tsv)
    s1 = (pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(is_unlabelled())
            .select(s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address").collect())
    rec = (pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select("entity_id", "business_name", "business_address", "country")
                      for k in (2, 3)]).filter(is_unlabelled())
             .select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address", country="country")
             .join(base.lazy().select("q").unique(), on="q", how="anti").collect())
    s1, rec = with_key(s1, "sa"), with_key(rec, "qa")
    log(f"same-address: {s1.height} S1 rows and {rec.height} unmatched records in countries without training labels; "
        f"with an address key {s1['k'].is_not_null().sum()} / {rec['k'].is_not_null().sum()} ({time.time() - t0:.0f}s)")
    c = unique_noise_pairs(rec, s1)
    p = c.filter("keep").drop("ok", "n_ok", "keep")
    log(f"same-address: {c.height} prefiltered same-address pairs, {c['ok'].sum()} with noise-only names, "
        f"{p.height} unique per record ({time.time() - t0:.0f}s)")
    n0 = p.height
    for path in exclude:
        p = p.join(pl.read_parquet(path, columns=["s", "q"]).select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64)), on=["s", "q"], how="anti")
    n1 = p.height
    if second_tsv:
        elsewhere = (p.select("s", "q").join(pairs(second_tsv).rename({"s": "s_second"}), on="q")
                      .filter(pl.col("s_second") != pl.col("s")).select("q").unique())
        p = p.join(elsewhere, on="q", how="anti")
    log(f"same-address: {n0 - n1} pairs in the excluded sets, {n1 - p.height} records the second pipeline matches elsewhere; {p.height} left")
    p = safe(street_twins(p, s1))
    n2 = p.height
    p = (p.join(base.group_by("s").len("s1_nmatch"), on="s", how="left").with_columns(pl.col("s1_nmatch").fill_null(0))
          .filter(pl.col("s1_nmatch") > 0))
    log(f"same-address: {n2} pass the safety filter, {p.height} with a non-empty S1 row "
        f"(acronym {p['acr'].sum()}, other {(~p['acr']).sum()}) ({time.time() - t0:.0f}s)")
    assert p["q"].n_unique() == p.height
    return p.select("s", "q", "country", "qn", "qa", "sn", "sa", "nops", "aops", "acr", "nb", "street_twins", "dom_exact",
                    "s1_nmatch").sort("s", "q")


def compare(new, ref):
    """(equal, pairs only in new, pairs only in ref): polars frame equality of the sorted distinct (s, q) pairs."""
    a = new.select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64)).unique().sort("s", "q")
    b = ref.select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64)).unique().sort("s", "q")
    return a.equals(b), a.join(b, on=["s", "q"], how="anti"), b.join(a, on=["s", "q"], how="anti")


def main():
    """Command line: build the pairs, write them to `out`, and with --ref compare them with a reference set (exit 1 if
    the pair sets differ)."""
    ap = argparse.ArgumentParser(description="Same-address pairs for a base submission (module docstring).")
    ap.add_argument("base_tsv")
    ap.add_argument("second_tsv")
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--ref")
    a = ap.parse_args()
    p = build(a.base_tsv, a.second_tsv, a.exclude)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    p.write_parquet(a.out)
    log(f"wrote {a.out}: {p.height} pairs")
    if a.ref:
        eq, only_new, only_ref = compare(p, pl.read_parquet(a.ref, columns=["s", "q"]))
        log(f"equal to {a.ref}: {eq}; only here {only_new.height}, only in the reference {only_ref.height}")
        sys.exit(0 if eq else 1)


if __name__ == "__main__":
    main()
