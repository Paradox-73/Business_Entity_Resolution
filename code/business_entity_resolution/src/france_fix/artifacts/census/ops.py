"""Detectors for the data generator's modification operations between a record (S2/S3) and an S1 row.

    from ops import ops
    ops(rec_name, rec_addr, s1_name, s1_addr) -> set of op names

Name ops start with 'n_', address ops with 'a_'. Word-level name ops carry the class of the changed word:
n_add:<cls>, n_drop:<cls>, n_swap:<cls>  with cls in legal / noise / title / loc / desc / other.
Works for US, India and France text (accents, French legal forms and street types included).
"""
import re
import unicodedata
from rapidfuzz import fuzz

# ---------------------------------------------------------------- word classes
LEGAL = set("""llc inc incorporated corp corporation co company ltd limited pvt private llp lp pllc pc plc public
sarl sas sasu sa eurl sci snc ei eirl selarl scp gie earl gmbh ag bv nv""".split())
LEGAL_CANON = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt"}
# generic words that the generator appears to add/drop as noise (US/India/France); refined from the census
NOISE = set("""group groupe services service holdings holding partners enterprises enterprise industries international
solutions global associates associes fils cie developpement france india usa us america trading and et
""".split())
TITLE = set("mr mrs ms dr smt sri shri m s messrs the le la les".split())
DESC = set("""amicale comite ecole club centre center association federation union syndicat societe cooperative
institute institut foundation fondation church eglise clinic clinique hospital hopital school college academy
academie restaurant cafe bar hotel pharmacie pharmacy bakery boulangerie garage auto studio salon""".split())
STOPW = set("of de des du d l a en and et the".split())

US_STATES = {"al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
             "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
             "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
             "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
             "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
             "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
             "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
             "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
             "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
             "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"}
IN_STATES = {"mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka", "tn": "tamil nadu",
             "wb": "west bengal", "gj": "gujarat", "tg": "telangana", "ts": "telangana", "ap": "andhra pradesh",
             "kl": "kerala", "rj": "rajasthan", "hr": "haryana", "pb": "punjab", "mp": "madhya pradesh", "br": "bihar",
             "or": "odisha", "od": "odisha", "jh": "jharkhand", "ct": "chhattisgarh", "cg": "chhattisgarh",
             "as": "assam", "uk": "uttarakhand", "hp": "himachal pradesh", "ga": "goa", "jk": "jammu and kashmir",
             "ch": "chandigarh", "py": "puducherry"}
FR_REGIONS = {"nouvelle aquitaine", "pays de la loire", "hauts de france", "ile de france", "auvergne rhone alpes",
              "provence alpes cote d azur", "occitanie", "grand est", "bretagne", "normandie", "bourgogne franche comte",
              "centre val de loire", "corse"}
STATE_FULL = set(US_STATES.values()) | set(IN_STATES.values()) | {"keralam", "orissa"} | FR_REGIONS
STATE_ABBR = set(US_STATES) | set(IN_STATES)
STREET = {"drive": "dr", "avenue": "ave", "street": "st", "road": "rd", "lane": "ln", "court": "ct", "circle": "cir",
          "boulevard": "blvd", "place": "pl", "terrace": "ter", "trail": "trl", "parkway": "pkwy", "highway": "hwy",
          "square": "sq", "cove": "cv", "point": "pt", "crossing": "xing", "heights": "hts", "mount": "mt",
          "north": "n", "south": "s", "east": "e", "west": "w", "floor": "fl", "building": "bldg", "suite": "ste",
          "apartment": "apt", "number": "no",
          "rue": "r", "boulevard_fr": "bd", "avenue_fr": "av", "chemin": "ch", "route": "rte", "impasse": "imp",
          "allee": "all", "place_fr": "pl", "faubourg": "fbg", "quai": "q", "square_fr": "sq", "residence": "res"}
STREET_ABBR = {v: k for k, v in STREET.items()} | {"av": "avenue", "bd": "boulevard", "str": "street", "rd.": "road"}
UNIT = set("unit fl floor bldg building suite ste apt apartment flat room block office door house h no".split())
CITY_AFFIX = set("city of town township village cdp borough county".split())
NULLTOK = re.compile(r"(?i)(?:^|,\s*)(?:null|none|n/a|na|<null>|-|\?)\s*(?=,|$)")


def strip_acc(x):
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()


def has_acc(x):
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFKD", x)) and strip_acc(x) != x


def non_latin(x):
    return any(ord(c) > 0x24F and c.isalpha() for c in x)


LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "7": "t", "8": "b", "@": "a", "$": "s"})


def words(x):
    """lowercase, accents stripped, punctuation -> space; '&' and '+' kept as words."""
    x = strip_acc(x).lower().replace("&", " & ").replace("+", " + ")
    return [w for w in re.split(r"[^a-z0-9&+]+", x) if w]


def undot_legal(x):
    """'L.L.C.' -> 'llc', 'Pvt.' -> 'pvt' (lowercased, accents stripped)."""
    x = strip_acc(x).lower()
    x = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), x)
    return x


def wclass(w):
    if w in LEGAL:
        return "legal"
    if w in NOISE or w in ("&", "+"):
        return "noise"
    if w in TITLE:
        return "title"
    if w in DESC:
        return "desc"
    if w in STATE_FULL or w in ("delhi", "mumbai", "bangalore", "kolkata", "chennai", "hyderabad", "pune", "paris"):
        return "loc"
    return "other"


def _match(a, b):
    return a == b or (len(a) > 2 and len(b) > 2 and fuzz.ratio(a, b) >= 75) or a.translate(LEET) == b.translate(LEET)


def _typo_kind(a, b):
    """kind of character edit turning S1 word a into record word b (both lowercase, a != b)."""
    if a.translate(LEET) == b.translate(LEET) and any(c.isdigit() for c in b) and not any(c.isdigit() for c in a):
        return "n_leet"
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]:
            return "n_typo_transpose"
        return "n_typo_subst" if len(d) == 1 else "n_typo_multi"
    if len(b) == len(a) + 1:
        return "n_typo_insert"
    if len(b) == len(a) - 1:
        return "n_typo_delete"
    return "n_typo_multi"


def name_ops(qn, sn):
    o = set()
    if not qn or not qn.strip():
        o.add("n_missing")
        return o
    sn = sn or ""
    if non_latin(qn):
        o.add("n_script")
        return o
    # ---- surface form
    ql = [c for c in qn if c.isalpha()]
    sl = [c for c in sn if c.isalpha()]
    if ql and qn == qn.lower() and sn != sn.lower():
        o.add("n_lower")
    elif ql and qn == qn.upper() and sn != sn.upper():
        o.add("n_upper")
    elif strip_acc(qn).lower() == strip_acc(sn).lower() and qn != sn and strip_acc(qn) != strip_acc(sn):
        o.add("n_case_other")
    elif ql and sl and qn == qn.title() and sn != sn.title() and qn.lower() == sn.lower():
        o.add("n_title")
    if has_acc(qn) and not has_acc(sn):
        o.add("n_acc_add")
    elif has_acc(sn) and not has_acc(qn):
        o.add("n_acc_strip")
    if "  " in qn and "  " not in sn:
        o.add("n_dblspace")
    if re.search(r"[\(\[\{]", qn) and not re.search(r"[\(\[\{]", sn):
        o.add("n_bracket")
    if re.search(r"\w-\w", qn) and not re.search(r"\w-\w", sn):
        o.add("n_hyphen")
    if qn.count(",") > sn.count(","):
        o.add("n_comma_add")
    elif qn.count(",") < sn.count(","):
        o.add("n_comma_drop")
    if re.search(r"#\d{3,}", qn) and not re.search(r"#\d{3,}", sn):
        o.add("n_idtag")
    if re.search(r"(?i)\.(com|net|org|fr|in|co)\b", qn) and not re.search(r"(?i)\.(com|net|org|fr|in)\b", sn):
        o.add("n_domain")
    if re.search(r"(?i)\btrading as\b|\bt/a\b|\bdba\b|\bd/b/a\b", qn) and not re.search(r"(?i)\btrading as\b|\bdba\b", sn):
        o.add("n_tradingas")
    # legal form dotted
    if re.search(r"\b(?:[A-Za-z]\.){2,}", qn) and not re.search(r"\b(?:[A-Za-z]\.){2,}", sn):
        o.add("n_legal_dot")
    elif re.search(r"(?i)\b(pvt|ltd|inc|corp|co)\.", qn) and not re.search(r"(?i)\b(pvt|ltd|inc|corp|co)\.", sn):
        o.add("n_legal_dot")
    if "n_domain" in o:                       # domain form implies lowercase + squashed + legal form dropped
        o.discard("n_lower")
        o.discard("n_comma_drop")
        return o
    # ---- word level
    qw, sw = words(undot_legal(qn)), words(undot_legal(sn))
    if not sw:
        o.add("n_s1_empty")
        return o
    # squashed: record has fewer words and its joined text equals S1 joined text
    js, jq = "".join(w for w in sw if w not in LEGAL), "".join(w for w in qw if w not in LEGAL)
    if len(qw) < len(sw) and any(len(w) > 8 for w in qw) and js and (js in jq or fuzz.ratio(js, jq) >= 85) and \
            len([w for w in qw if w not in LEGAL]) < len([w for w in sw if w not in LEGAL]):
        o.add("n_squash")
        qw = sw[:] if js == jq else qw
        if "n_squash" in o and js == jq:
            pass
    # acronym
    core = [w for w in sw if w not in LEGAL and w not in STOPW]
    if len(core) >= 2 and any(w == "".join(x[0] for x in core) for w in qw) and len(qw) <= 2:
        o.add("n_acronym")
    # & / and / + swaps
    amp = lambda ws: {w for w in ws if w in ("&", "+", "and", "et")}
    if amp(qw) and amp(sw) and amp(qw) != amp(sw):
        o.add("n_amp_swap")
    # canonical legal forms
    def canon(ws):
        out = []
        for i, w in enumerate(ws):
            c = LEGAL_CANON.get(w, w)
            out.append(c)
        return out
    qc, sc = canon(qw), canon(sw)
    ql_, sl_ = [w for w in qc if w in LEGAL], [w for w in sc if w in LEGAL]
    if [w for w in qw if w in LEGAL] != [w for w in sw if w in LEGAL] and ql_ == sl_:
        o.add("n_legal_abbrev")                     # Private Limited <-> Pvt Ltd, Company <-> Co
    lq, ls = set(ql_), set(sl_)
    if ls and lq and not (ls & lq):
        o.add("n_legal_change")
    elif ls and lq and ls != lq:
        o.add("n_legal_partial")
    elif ls and not lq:
        o.add("n_legal_drop")
    elif lq and not ls:
        o.add("n_legal_add")
    qx = [w for w in qc if w not in LEGAL]
    sx = [w for w in sc if w not in LEGAL]
    # duplicates
    if any(qx.count(w) > sx.count(w) and w in sx for w in set(qx)):
        o.add("n_word_dup")
    # match words (fuzzy)
    used = [False] * len(qx)
    unmatched_s = []
    order = []
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None:
            j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
            if j is not None:
                o.add(_typo_kind(a, qx[j]))
        if j is None:
            unmatched_s.append(a)
        else:
            used[j] = True
            order.append(j)
    unmatched_q = [qx[j] for j in range(len(qx)) if not used[j]]
    if order != sorted(order):
        o.add("n_word_order")
    if "n_squash" in o or "n_acronym" in o:
        return o
    if "n_word_dup" in o:
        unmatched_q = [w for w in unmatched_q if w not in sx]
    k = min(len(unmatched_s), len(unmatched_q))
    for a, b in zip(unmatched_s[:k], unmatched_q[:k]):
        ca, cb = wclass(a), wclass(b)
        o.add("n_swap:" + (ca if ca == cb else f"{ca}>{cb}" if "other" not in (ca, cb) else ("other" if ca == cb else (cb if ca == "other" else ca))))
    for a in unmatched_s[k:]:
        o.add("n_drop:" + wclass(a))
    for b in unmatched_q[k:]:
        o.add("n_add:" + wclass(b))
    if not qx and sx:
        o.add("n_only_legal")
    return o


# ---------------------------------------------------------------- address
def _comps(a):
    return [c.strip() for c in a.split(",") if c.strip()]


def _cnorm(c):
    return " ".join(words(c))


def house(a, country=None):
    """first house number (digits only, leading zeros stripped), raw token, prefix flag, zero-pad flag, end offset.
    France: 5-digit tokens are postcodes, not house numbers, and are skipped."""
    x = strip_acc(a).lower()
    if country == "France":
        x = re.sub(r"(?<![0-9])\d{5}(?![0-9])", " ", x)
    m = re.search(r"(?<![0-9a-z])(#+|no\.?\s*#?|n[o°º]\s*|h\.?\s*no\.?\s*#?|num(?:ero)?\.?\s*)?(0*)(\d{1,6})(?!\d|st\b|nd\b|rd\b|th\b)([a-z]?)(?![0-9])", x)
    if not m:
        return None
    return m.group(3), m.group(0), bool(m.group(1)), bool(m.group(2)), m.end()


def addr_ops(qa, sa, country=None):
    o = set()
    if qa is None or not str(qa).strip() or str(qa).strip().lower() in ("none", "null", "nan"):
        return {"a_missing"}
    sa = sa or ""
    if non_latin(qa) and not non_latin(sa):
        o.add("a_script")
    if re.search(r"(?i)(?:^|,\s*)(?:null|n/a|<null>|none)\s*(?=,|$)", qa):
        o.add("a_nulltok")
    if qa == qa.upper() and sa != sa.upper() and any(c.isalpha() for c in qa):
        o.add("a_upper")
    elif qa == qa.lower() and sa != sa.lower() and any(c.isalpha() for c in qa):
        o.add("a_lower")
    if re.search(r"(?i)\bp\.?o\.? box\b", qa) and not re.search(r"(?i)\bp\.?o\.? box\b", sa):
        o.add("a_pobox")
    if has_acc(qa) and not has_acc(sa):
        o.add("a_acc_add")
    elif has_acc(sa) and not has_acc(qa):
        o.add("a_acc_strip")
    cs, cq = [_cnorm(c) for c in _comps(sa)], [_cnorm(c) for c in _comps(qa)]
    cq = [c for c in cq if c not in ("null", "n a", "none", "")]
    # number ops
    hs, hq = house(sa, country), house(qa, country)
    if hs and not hq:
        o.add("a_num_missing")
    elif hq and not hs:
        o.add("a_num_add")
    elif hs and hq:
        if hq[2] and not hs[2]:
            o.add("a_num_prefix")
        if hq[3] and not hs[3]:
            o.add("a_num_pad")
        if hs[0] != hq[0]:
            try:
                d = int(hq[0]) - int(hs[0])
            except ValueError:
                d = 0
            if hq[0].startswith(hs[0]) or hs[0].startswith(hq[0]):
                o.add("a_num_prefixmatch")      # 25 -> 25/7 read as 257? or 446 -> 446-448
            ad = abs(d)
            o.add("a_num_" + ("up" if d > 0 else "down") + ("1" if ad <= 1 else "2_5" if ad <= 5 else "6_20" if ad <= 20 else "big"))
        else:
            rs = strip_acc(sa).lower()[hs[4]:hs[4] + 3]
            rq = strip_acc(qa).lower()[hq[4]:hq[4] + 3]
            if re.match(r"\s*[/\-]\s*\d", rq) and not re.match(r"\s*[/\-]\s*\d", rs):
                o.add("a_num_suffix")          # 25 -> 25/7, 446 -> 446-448
            if re.match(r"\.", rq) and not re.match(r"\.", rs):
                o.add("a_num_dot")
    ns = re.findall(r"\d+", strip_acc(sa))
    nq = re.findall(r"\d+", strip_acc(qa))
    if len(nq) > len(ns) and "a_num_suffix" not in o:
        o.add("a_extra_numbers")
    # component-level
    ws, wq = set(words(sa)), set(words(qa))
    for full, ab in STREET.items():
        full = full.split("_")[0]
        if full in ws and full not in wq and ab in wq and ab not in ws:
            o.add("a_street_abbr")
        if ab in ws and ab not in wq and full in wq and full not in ws:
            o.add("a_street_expand")
    stt = {**US_STATES, **IN_STATES}
    for ab, full in stt.items():
        if (ab in cs and full in cq) and not (full in cs):
            o.add("a_state_expand")
        if (full in cs and ab in cq) and not (ab in cs):
            o.add("a_state_abbr")
    if (country == "India" or not country) and non_latin(qa):
        o.add("a_state_native")
    unit_s = any(w in UNIT for w in ws)
    unit_q = any(w in UNIT for w in wq)
    if unit_s and not unit_q:
        o.add("a_unit_drop")
    elif unit_q and not unit_s:
        o.add("a_unit_add")
    # component matching
    def cmatch(a, b):
        return a == b or fuzz.ratio(a, b) >= 85 or (stt.get(a) == b) or (stt.get(b) == a) or \
            fuzz.token_set_ratio(a, b) >= 90
    used = [False] * len(cq)
    order, miss_s = [], []
    for i, a in enumerate(cs):
        j = next((j for j, b in enumerate(cq) if not used[j] and cmatch(a, b)), None)
        if j is None:
            miss_s.append(a)
        else:
            used[j] = True
            order.append(j)
            if a != cq[j] and fuzz.ratio(a, cq[j]) >= 85 and not (stt.get(a) == cq[j] or stt.get(cq[j]) == a):
                aw, bw = set(a.split()), set(cq[j].split())
                if (bw - aw) & CITY_AFFIX or (aw - bw) & CITY_AFFIX:
                    o.add("a_city_affix")
                elif re.sub(r"[\d\s]+", "", a) == re.sub(r"[\d\s]+", "", cq[j]):
                    pass
                elif not any(STREET.get(x) in bw or STREET_ABBR.get(x) in bw for x in aw - bw):
                    o.add("a_typo")
            elif a != cq[j] and fuzz.token_set_ratio(a, cq[j]) >= 90 and len(cq[j]) < len(a) and                     re.sub(r"[\d\s]+", "", a) != re.sub(r"[\d\s]+", "", cq[j]):
                o.add("a_comp_trunc")
    miss_q = [cq[j] for j in range(len(cq)) if not used[j]]
    if order != sorted(order):
        o.add("a_shuffle")
    def ckind(c):
        if c in STATE_FULL or c in STATE_ABBR:
            return "state"
        if set(c.split()) & (UNIT - {"no", "h"}):
            return "unit"
        if re.search(r"\d", c):
            return "street"
        if set(c.split()) & (set(STREET) | set(STREET_ABBR)):
            return "street"
        return "place"
    ks, kq = [ckind(c) for c in miss_s], [ckind(c) for c in miss_q]
    for k in set(ks):
        if k not in kq:
            o.add("a_drop:" + k)
    for k in set(kq):
        if k not in ks:
            o.add("a_add:" + k)
    for k in set(ks) & set(kq):
        o.add("a_chg:" + k)
    return o


def ops(rec_name, rec_addr, s1_name, s1_addr, country=None):
    return name_ops(rec_name, s1_name) | addr_ops(rec_addr, s1_addr, country)


if __name__ == "__main__":
    tests = [("CARPENTERS LOCAL 438 438", "5299 Royal Arh Cascade Drive, Bldg 45, Dublin, Ohio", "Carpenters Local 438",
              "5299 Royal Arch Cascade Drive, Bldg 45, Dublin, OH"),
             ("Sbiuper 11 Pub Inc", "MN, BURNSVILLE, 13009 IRVING AVE", "Super 11 Pub Inc", "13009 Irving Avenue, MN, Burnsville"),
             ("unitedfoundation.com", "95TH STREET, CLEVELAND, OH", "United Foundation Inc", "2082 95th Street, Cleveland, OH"),
             ("National Green Games Ltd", "325 Napoleon Ave, Columbus, Ohio", "National Green Games Inc.", "321 Napoleon Avenue, Columbus, OH"),
             ("Zenith Seaf0od LLC", "5 BALDWIN ROAD, WHITESTOWN, NY", "Zenith Seafood Inc", "2 Baldwin Road, Whitestown, NY"),
             ("Mr Sanjeet Mining Private Ltd", "OFFICE NO 25/7, A & B, VIGHNAHARA BUILDING, PUNE, Maharashtra",
              "Sanjeet Mining Private Limited", "Office No 25, A & B, Vighnahara Building, Pune, Maharashtra"),
             ("Physical Partners Platinum Therapy LLC", "530. STATE ST, WESTERVILLE, OH", "Physical Therapy Platinum Partners LLC",
              "530 State Street, Unit 205W, Westerville, OH"),
             ("SUMMITPRAIRIERIVERNORTH.COM", None, "Summit Prairie Rivernorth, LLC", "647 Gurdev Circle, City Of Socorro, TX")]
    for t in tests:
        print(t[0], "|", t[2], "->", sorted(ops(*t)))
