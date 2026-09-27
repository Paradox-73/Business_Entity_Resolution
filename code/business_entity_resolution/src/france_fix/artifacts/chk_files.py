import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import ROOT  # noqa: E402
from collections import Counter
import polars as pl
from common import WORK
s1 = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"])
c = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
def rd(p): return dict(l.rstrip("\n").split("\t") for l in open(p, encoding="utf8"))
def npairs(d, cty): return sum(len(v.split(",")) for k, v in d.items() if c.get(k) == cty and v)
R = f"{ROOT}/"
base = {"v7ens": rd(R + "submissions/v7ens/matching_results.tsv"), "v7p": rd(R + "submissions/v7p/matching_results.tsv")}
for name, path, main in [(a, b, m) for a, b, m in (x.split("|") for x in sys.argv[1:])]:
    B = rd(R + path); A = base[main]
    diff = Counter(c.get(k, "hdr") for k in A if A[k] != B.get(k))
    print(f"{name}: vs {main} differing rows {dict(diff)}; France pairs {npairs(B, 'France')}; US {npairs(B, 'US')} (main {npairs(A, 'US')}); India {npairs(B, 'India')} (main {npairs(A, 'India')})")
