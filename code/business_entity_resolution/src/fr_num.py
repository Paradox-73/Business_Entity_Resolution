"""France house-number veto: reject a France pair when both addresses carry a street number and the numbers differ.

Evidence (LB, 26 Sep): v7j added 21k France records whose house number differs from the S1 row (the model's best
ranked ones) and France fell by 0.0163 (LB 0.98071 vs 0.983159) -> in France, number-changed records are fakes. v7i
(transformer additions, 34% number-changed) also lost (France -0.0035). So France's generator keeps the house number
of true records (unlike US/India, where 11-20% of true records change it; that is why the models, trained on US/India,
accept some France number changes). Same logic as the legal-form veto.

Not vetoed: one number is the start of the other ("175" vs "1", "103" vs "10"; cut numbers are a known noise).
The street number is the number written right before the street type (rue, avenue, bd, ...), so apartment, floor,
postal-box and building numbers are ignored; "bis/ter/b" and leading zeros are ignored. No street number on either
side -> no veto.

  python fr_num.py <scores.parquet> <out.parquet> [min_p=0.05]
"""
import re
import sys
import unicodedata
import polars as pl
from common import WORK, log, id_to_int

STREET = (r"rue|r|avenue|av|ave|aven|boulevard|bd|blvd|boul|allee|allees|all|chemin|che|ch|place|pl|impasse|imp|route|rte|"
          r"quai|cours|crs|square|sq|passage|pass|voie|sentier|sente|promenade|prom|esplanade|parvis|rond|lotissement|lot|"
          r"hameau|ham|residence|res|faubourg|fbg|mail|cite|villa|port|zone|za|zi|zac|cour|carrefour|clos|parc|domaine|"
          r"montee|descente|traverse|venelle|ruelle|galerie|chaussee|levee|digue|rocade|periph")
NUM_RE = re.compile(r"(?<![0-9])0*(\d{1,5})\s*(?:bis|ter|quater|[a-d])?\s*[,.\-]?\s*(?:" + STREET + r")\b")


def street_num(a):
    x = unicodedata.normalize("NFKD", a or "").encode("ascii", "ignore").decode().lower()
    x = re.sub(r"(?:n\s*[o°º]|no\.?|num(?:ero)?\.?|#)\s*(?=\d)", " ", x)       # "N° 30", "No. 30", "#30"
    m = NUM_RE.search(x)
    return m.group(1) if m else None


def main(src, out, min_p=0.05):
    b = pl.read_parquet(src)
    s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "business_address", "country"]).filter(
        pl.col("country") == "France").select(s=id_to_int("entity_id"), sa="business_address")
    cand = b.filter(pl.col("p2") >= min_p).select("q", "s", "p2").join(s1, on="s")
    qa = pl.concat([pl.scan_parquet(f"{WORK}/test_s{k}.parquet").select(q=id_to_int("entity_id"), qa="business_address")
                    .join(cand.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
    cand = cand.join(qa, on="q")
    ns = [street_num(x) for x in cand["sa"].to_list()]
    nq = [street_num(x) for x in cand["qa"].to_list()]
    cand = cand.with_columns(ns=pl.Series(ns, dtype=pl.Utf8), nq=pl.Series(nq, dtype=pl.Utf8))
    # numbers differ, and neither is the start of the other ("175" vs "1" can be a cut number, a known noise of true records)
    veto = cand.filter(pl.col("ns").is_not_null() & pl.col("nq").is_not_null() & (pl.col("ns") != pl.col("nq"))
                       & ~pl.col("ns").str.starts_with(pl.col("nq")) & ~pl.col("nq").str.starts_with(pl.col("ns")))
    log(f"France pairs checked (p2 >= {min_p}): {cand.height}; both street numbers found: "
        f"{cand.filter(pl.col('ns').is_not_null() & pl.col('nq').is_not_null()).height}; vetoed (numbers differ): {veto.height} "
        f"({veto.filter(pl.col('p2') >= 0.5).height} with p2 >= 0.5)")
    for r in veto.filter(pl.col("p2") >= 0.5).sample(n=min(12, veto.filter(pl.col("p2") >= 0.5).height), seed=1).iter_rows(named=True):
        log(f"   {r['ns']:>5} vs {r['nq']:>5} | {(r['sa'] or '')[:55]:55s} || {(r['qa'] or '')[:55]}")
    b = b.join(veto.select("q", "s", v=pl.lit(True)), on=["q", "s"], how="left").with_columns(
        p2=pl.when(pl.col("v")).then(0.0).otherwise(pl.col("p2")).cast(b["p2"].dtype)).drop("v")
    b.write_parquet(out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
