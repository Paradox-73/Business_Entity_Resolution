"""France pairs of earlier files -> $BER_SCRATCH/france/pairs_<version>.parquet (v7ens, v7g, v7g_num, v7j, v7i, v7k),
then the France F0.5 change between pairs of those files under several hypotheses about which changed pairs are true,
printed next to the leaderboard result where one exists. Reads fr_top.parquet of fr1.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
from common import ROOT  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int
OUT = f"{SCRATCH}/france"
R = f"{ROOT}/"
NFR = 259452
s1t = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(s=id_to_int("entity_id"), c="country")
frs = s1t.filter(pl.col("c") == "France").select("s")
def pairs(p):
    sub = pl.read_csv(R + p, separator="\t", quote_char=None, infer_schema_length=0)
    return (sub.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(s=id_to_int("source1_entity_id"), q=id_to_int("matched_entity_ids")).join(frs, on="s"))
top = pl.read_parquet(f"{OUT}/fr_top.parquet").select("q", "s", "pat")
files = {"v7ens": "submissions/v7ens/matching_results.tsv", "v7g": "work/out_v7g/matching_results.tsv", "v7g_num": "submissions/v7g_num/matching_results.tsv",
         "v7j": "submissions/v7j/matching_results.tsv", "v7i": "submissions/v7i/matching_results.tsv", "v7k": "submissions/v7k/matching_results.tsv"}
P = {k: pairs(v) for k, v in files.items()}
for k, v in P.items(): v.write_parquet(f"{OUT}/pairs_{k}.parquet")
def f05(tp, npred, ntrue):
    if npred == 0 and ntrue == 0: return 1.0
    if tp == 0: return 0.0
    p, r = tp / npred, tp / ntrue
    return 1.25 * p * r / (0.25 * p + r)
def delta(X, Y, htrue):
    """rows touched; unchanged pairs assumed true; changed pairs true iff htrue(pat)."""
    rem = X.join(Y, on=["q", "s"], how="anti").with_columns(side=pl.lit("rem"))
    add = Y.join(X, on=["q", "s"], how="anti").with_columns(side=pl.lit("add"))
    ch = pl.concat([rem, add]).join(top, on=["q", "s"], how="left").with_columns(pl.col("pat").fill_null("notbest|x"))
    ch = ch.with_columns(t=pl.col("pat").map_elements(htrue, return_dtype=pl.Boolean))
    nx = X.group_by("s").agg(nx=pl.len())
    g = ch.group_by("s").agg(rem_t=((pl.col("side") == "rem") & pl.col("t")).sum(), rem_f=((pl.col("side") == "rem") & ~pl.col("t")).sum(),
                             add_t=((pl.col("side") == "add") & pl.col("t")).sum(), add_f=((pl.col("side") == "add") & ~pl.col("t")).sum()).join(nx, on="s", how="left").with_columns(pl.col("nx").fill_null(0))
    tot = 0.0
    for s, rt, rf, at, af, n in g.iter_rows():
        unchanged = n - rt - rf
        ntrue = unchanged + rt + at
        before = f05(unchanged + rt, n, ntrue)
        after = f05(unchanged + at, unchanged + at + af, ntrue)
        tot += after - before
    return tot / NFR, g.height, rem.height, add.height
H = {
 "numdiff fake, rest true": lambda p: not p.endswith("|ndiff"),
 "all changed true": lambda p: True,
 "all changed fake": lambda p: False,
 "ndiff fake, nsame true, nmiss/other half": lambda p: p.endswith("|nsame"),
}
for a, b, lb in (("v7ens", "v7j", -0.0163), ("v7ens", "v7i", -0.0035), ("v7ens", "v7g", None), ("v7g", "v7g_num", None), ("v7ens", "v7g_num", "-0.0022..-0.0004"), ("v7ens", "v7k", None)):
    for hn, h in H.items():
        d, rows, nr, na = delta(P[a], P[b], h)
        print(f"{a}->{b} (LB France {lb}) rows {rows} removed {nr} added {na} | H[{hn}]: dFR {d:+.4f}")
