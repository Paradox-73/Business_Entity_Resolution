"""Learn address normalisation tables from the TRAINING matched pairs only (no outside data).

1. abbrev_map: short token -> long form, e.g. "rd" -> "road", "tx" -> "texas", "nc" -> "north carolina".
   A pair (short, long) is counted when, in a true matched pair, `short` appears only in one address
   and `long` only in the other, `short` starts with the same letter and is a subsequence of `long`
   (or equals the initials of 2 consecutive tokens). Kept when frequent and dominant.
2. script_map: local-script phrase -> latin text, e.g. "महाराष्ट्र" -> "maharashtra".
   Counted as (non-latin run in one address, comma field of the S1 address missing from the other).
Output: WORK/maps.json
"""
import json
import os
import re
from collections import Counter, defaultdict
import polars as pl
from common import WORK, read_tsv, raw_path, read_truth, log
from normalize import _ascii_lower, _JUNK, _squash

N_PAIRS = 400_000


def is_subseq(s, t):
    """True if the characters of s appear in t in the same order."""
    it = iter(t)
    return all(c in it for c in s)


def base_addr(e):
    """Polars expression: address lowercased, accents and non-ASCII characters removed, punctuation turned
    into spaces, junk tokens dropped."""
    x = _ascii_lower(e).str.replace_all(r"[^\x00-\x7F]", " ").str.replace_all(r"[^a-z0-9]+", " ")
    return _squash(x.str.replace_all(_JUNK, " "))


def main():
    """Learn abbrev_map and script_map from 400,000 sampled true train pairs (rules in the module docstring) and
    write WORK/maps.json."""
    gt = read_truth().sample(N_PAIRS, seed=0)
    s1 = read_tsv(raw_path("train", 1)).filter(pl.col("entity_id").is_in(gt["s1_id"].unique().to_list()))
    need = gt["q_id"].to_list()
    q = pl.concat([read_tsv(raw_path("train", k)).filter(pl.col("entity_id").is_in(need)) for k in (2, 3)])
    pr = (gt.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_address").alias("a1")), on="s1_id")
            .join(q.select(pl.col("entity_id").alias("q_id"), pl.col("business_address").alias("a2")), on="q_id")
            .with_columns(b1=base_addr(pl.col("a1")), b2=base_addr(pl.col("a2"))))
    log("pairs for map learning:", pr.height)

    cnt = Counter()
    short_tot = Counter()
    script = defaultdict(Counter)
    for a1, a2, b1, b2 in pr.select("a1", "a2", "b1", "b2").iter_rows():
        t1, t2 = b1.split(), b2.split()
        s1set, s2set = set(t1), set(t2)
        for A, B, tl in ((s2set, s1set, t1), (s1set, s2set, t2)):
            onlyA = [x for x in A - B if x.isalpha() and len(x) <= 5]
            onlyB = [y for y in B - A if y.isalpha()]
            for x in onlyA:
                short_tot[x] += 1
                for y in onlyB:
                    if len(y) > len(x) and y[0] == x[0] and is_subseq(x, y):
                        cnt[(x, y)] += 1
                if len(x) == 2:
                    for i in range(len(tl) - 1):
                        if tl[i][0] == x[0] and tl[i + 1][0] == x[1] and tl[i] in B - A and tl[i + 1] in B - A:
                            cnt[(x, tl[i] + " " + tl[i + 1])] += 1
        # local-script phrases in the candidate address
        a2 = a2 or ""
        for run in re.findall(r"[^\x00-\x7F][^\x00-\x7F ]*(?: [^\x00-\x7F][^\x00-\x7F ]*)*", a2):
            b2l = b2
            for f in (a1 or "").split(","):
                fl = re.sub(r"[^a-z0-9 ]+", " ", f.lower()).strip()
                if fl and fl not in b2l and not any(ch.isdigit() for ch in fl):
                    script[run][fl] += 1

    best = {}
    for (x, y), c in cnt.items():
        if c >= 15 and (x not in best or c > best[x][1]):
            best[x] = (y, c)
    abbrev = {x: y for x, (y, c) in best.items() if c / max(short_tot[x], 1) >= 0.3}
    smap = {}
    for run, ctr in script.items():
        (f, c), tot = ctr.most_common(1)[0], sum(ctr.values())
        if c >= 5 and c / tot >= 0.5:
            smap[run] = " " + f + " "
    json.dump({"abbrev": abbrev, "script": smap}, open(os.path.join(WORK, "maps.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=0)
    log(f"learned {len(abbrev)} abbreviations, {len(smap)} script phrases")
    log("sample abbrev:", list(abbrev.items())[:40])
    log("sample script:", list(smap.items())[:15])


if __name__ == "__main__":
    main()
