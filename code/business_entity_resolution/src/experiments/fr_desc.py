"""France descriptor-swap veto: reject a France pair when the record keeps the S1's other name words but REPLACES its
descriptor word (club, ecole, amicale, comite, ...) with a different descriptor or a company suffix (et fils,
& associes, groupe, ...).

Evidence (26 Sep, label-free): France S1 names are [brand/city] + [descriptor from a ~90-word list] + [legal form];
the France look-alike records keep brand and address and swap the descriptor ("Calais Amicale" -> "Calais Services",
"KID Comite SARL" -> "KID SARL Groupe"). 5.4% of v7ens's France matched pairs have this pattern. True records seen in
samples only have typos, accents, reordering, abbreviations. No France labels exist, so the effect is measured on the LB
(US/India untouched): France change = LB change / 0.15.

  python fr_desc.py <scores.parquet> <out.parquet> [min_p=0.2]
Only France pairs with p2 >= min_p are checked (lower ones cannot be chosen anyway).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/: shared modules
import re
import sys
import unicodedata
import polars as pl
from rapidfuzz import fuzz, process
from common import WORK, log, id_to_int

STOP = set("sarl sas sasu sa eurl sci snc ei eirl selarl scp gie earl inc llc ltd corp co company pvt private limited "
           "llp plc the and of de des du la le les et d l a s e r u www com net org in fr".split())
DESC = sorted(set("""club ecole amicale comite maison centre union sportive college amis primaire federation pharmacie cie
sante etablissements parents ets freres lycee fils maternelle societe association fetes institut anciens compagnie ehpad
loisirs culturelle clinique jeunes sport groupement medical elementaire hospitalier foyer culture section deleves groupe
collectif service services team chasse gestion residence danse patrimoine musique communale theatre cercle conseil atelier
publique associes holding participations bureau academie ateliers espace entreprise cabinet agence studio""".split()))
SAME = [{"service", "services"}, {"ets", "etablissements"}, {"atelier", "ateliers"}, {"sport", "sportive"}]


def toks(x):
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    return [w for w in re.sub(r"[^a-z0-9]+", " ", x).split() if w not in STOP]


_cache = {}


def desc_of(ws):
    out = set()
    for w in ws:
        if w not in _cache:
            if w in DESC:
                _cache[w] = w
            elif len(w) >= 4:
                m = process.extractOne(w, DESC, scorer=fuzz.ratio, score_cutoff=80)
                _cache[w] = m[0] if m else None
            else:
                _cache[w] = None
        if _cache[w]:
            out.add(_cache[w])
    return out


def desc_swap(s, q):
    ds, dq = desc_of(toks(s)), desc_of(toks(q))
    for grp in SAME:
        if grp & ds and grp & dq:
            ds, dq = ds - grp, dq - grp
    return bool(ds - dq) and bool(dq - ds)


def main(src, out, min_p=0.2):
    b = pl.read_parquet(src)
    s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "business_name", "country"]).filter(
        pl.col("country") == "France").select(s=id_to_int("entity_id"), sn="business_name")
    cand = b.filter(pl.col("p2") >= min_p).join(s1, on="s")
    qn = pl.concat([pl.scan_parquet(f"{WORK}/test_s{k}.parquet").select(q=id_to_int("entity_id"), qn="business_name")
                    .join(cand.select("q").unique().lazy(), on="q").collect() for k in (2, 3)])
    cand = cand.join(qn, on="q")
    flag = [desc_swap(a, c) for a, c in zip(cand["sn"].to_list(), cand["qn"].to_list())]
    veto = cand.filter(pl.Series(flag)).select("q", "s", v=pl.lit(True))
    log(f"France pairs checked (p2 >= {min_p}): {cand.height}; descriptor swaps vetoed: {veto.height} "
        f"({veto.join(b, on=['q', 's']).filter(pl.col('p2') >= 0.5).height} with p2 >= 0.5)")
    b = b.join(veto, on=["q", "s"], how="left").with_columns(
        p2=pl.when(pl.col("v")).then(0.0).otherwise(pl.col("p2")).cast(b["p2"].dtype)).drop("v")
    b.write_parquet(out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.2)
