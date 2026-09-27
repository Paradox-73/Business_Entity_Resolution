"""France: the transformer may not lower a pair whose record keeps the S1's house number AND street.

Evidence (27 Sep, label-free, France test best candidates; US/India labels for the decoy mechanism):
- Decoys shift the house number UP (US/India train); France records that add a pure decoy word (holding,
  international, distribution, participations) keep the number 3.1-3.3% of the time, records that add true-noise
  words (com, fils, et, cie, services, associes) 75-95%. Descriptor words (club, ecole, amicale, comite, sportive,
  college, pharmacie...) sit at 35-41%: a mix whose same-number part is only ~5% decoys (groupe/france/
  developpement at 21%: ~12% decoys among same-number records).
- The transformer lowers France same-number pairs 20x more often than US ones; it lowered 20.8k France same-number
  pairs that the GBDT scores >= 0.5 below the accept line (club 2.4k, groupe 2.7k, developpement 1.9k, ...).
- LB: v7g_num (more transformer lowering + number veto) lost France; v7i (transformer ADDITIONS) lost France, so
  only the lowering is undone here, never an addition beyond the GBDT p2.

  python fr_restore.py <scores_frmin.parquet> <gbdt_scores.parquet> <out.parquet> [min_g=0.5]
Rows: France pairs with equal first house number, same street (token-sort ratio >= 85 of the street text after the
number) and GBDT p2 >= min_g get p2 = max(p2, GBDT p2). Everything else is unchanged.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import re
import sys
import unicodedata
import polars as pl
from rapidfuzz import fuzz
from common import WORK, log, id_to_int

TYPES = {"rue", "r", "avenue", "av", "ave", "aven", "boulevard", "bd", "blvd", "boul", "allee", "allees", "all", "chemin",
         "che", "ch", "place", "pl", "impasse", "imp", "route", "rte", "quai", "cours", "crs", "square", "sq", "passage",
         "pass", "voie", "sentier", "promenade", "esplanade", "residence", "res", "faubourg", "fbg", "bis", "ter", "b",
         "de", "la", "le", "les", "du", "des", "d", "l", "et"}


def street(a):
    """(first house number, street words after it up to the next comma) of an address; (None, '') if no number."""
    x = unicodedata.normalize("NFKD", a or "").encode("ascii", "ignore").decode().lower()
    x = re.sub(r"(?:n\s*[o°º]|no\.?|num(?:ero)?\.?|#)\s*(?=\d)", " ", x)
    m = re.search(r"(?<![0-9])0*(\d{1,5})(?![0-9])", x)
    if not m:
        return None, ""
    rest = x[m.end():].split(",")[0]
    return m.group(1), " ".join(w for w in re.findall(r"[a-z]+", rest) if w not in TYPES)


def main(src, gbdt, out, min_g=0.5):
    b = pl.read_parquet(src)
    s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "business_address", "country"]).filter(
        pl.col("country") == "France").select(s=id_to_int("entity_id"), sa="business_address")
    g = pl.read_parquet(gbdt, columns=["q", "s", "p2"]).rename({"p2": "g"})
    cand = (b.select("q", "s", "p2").join(s1, on="s").join(g, on=["q", "s"])
            .filter((pl.col("g") >= min_g) & (pl.col("p2") < pl.col("g"))))
    qa = pl.concat([pl.scan_parquet(f"{WORK}/test_s{k}.parquet").select(q=id_to_int("entity_id"), qa="business_address")
                    .join(cand.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
    cand = cand.join(qa, on="q")
    ss = [street(a) for a in cand["sa"].to_list()]
    qs = [street(a) for a in cand["qa"].to_list()]
    ok = [bool(a[0] is not None and a[0] == c[0] and a[1] and c[1] and fuzz.token_sort_ratio(a[1], c[1]) >= 85)
          for a, c in zip(ss, qs)]
    fix = cand.filter(pl.Series(ok))
    log(f"France pairs lowered by the transformer below GBDT p2 >= {min_g}: {cand.height}; same house number and street: "
        f"{fix.height} ({fix.filter(pl.col('p2') < 0.5).height} crossing 0.5)")
    for r in fix.filter(pl.col("p2") < 0.5).sample(n=min(10, fix.height), seed=2).iter_rows(named=True):
        log(f"   p {r['p2']:.2f} -> {r['g']:.2f} | {(r['sa'] or '')[:48]:48s} || {(r['qa'] or '')[:48]}")
    b = b.join(fix.select("q", "s", "g"), on=["q", "s"], how="left").with_columns(
        p2=pl.when(pl.col("g").is_not_null()).then(pl.col("g")).otherwise(pl.col("p2")).cast(b["p2"].dtype)).drop("g")
    b.write_parquet(out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
