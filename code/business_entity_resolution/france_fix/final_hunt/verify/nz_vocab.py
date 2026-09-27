import os, sys
from collections import Counter
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from common import WORK
from ops import words, undot_legal, LEGAL, DESC
names = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).filter(pl.col("country") == "France").select("business_name").collect()["business_name"].to_list()
N = len(names)
cnt, anyw = Counter(), Counter()
for n in names:
    ws = [w for w in words(undot_legal(n or "")) if w not in LEGAL]
    for w in set(ws): anyw[w] += 1
    if len(ws) >= 2: cnt[ws[-1]] += 1
print("France S1", N)
print("last core word, top 45:", [(w, c) for w, c in cnt.most_common(45)])
for w in ["services", "service", "groupe", "france", "cie", "compagnie", "fils", "freres", "associes", "et", "amicale", "club", "comite", "ecole", "maison", "societe"]:
    print(f"{w:10s} any-position {anyw[w]:6d} ({anyw[w]/N:.4f})  last-core {cnt[w]:6d}")
