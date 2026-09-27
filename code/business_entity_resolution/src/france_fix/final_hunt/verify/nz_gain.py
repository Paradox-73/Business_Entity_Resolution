import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
from common import WORK, id_to_int, read_tsv
V = rf"{SCRATCH}/final/verify"
# gathik keeps France descriptor-veto pairs?
def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
g = pairs(os.path.join(WORK, "gathik", "v8_matching_results.tsv"))
veto = pl.read_parquet(rf"{SCRATCH}/frfix/namechg/veto_set.parquet", columns=["q", "s"]).with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
print("descriptor-veto pairs", veto.height, "kept by gathik:", veto.join(g, on=["s", "q"]).height)
gp = pl.scan_parquet(os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet")).select("q", "s", "p2").with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64)).join(veto.lazy(), on=["q", "s"]).collect()
print("gathik p on descriptor-veto pairs: n", gp.height, "share p>=0.9", (gp["p2"] >= 0.9).mean(), "mean", gp["p2"].mean())

d = pl.read_parquet(os.path.join(V, "nz_477.parquet"))
def F(tp, np_, nt):
    if nt == 0: return 1.0 if np_ == 0 else 0.0
    if tp == 0: return 0.0
    p, r = tp / np_, tp / nt
    return 1.25 * p * r / (0.25 * p + r)
def row_gain(m, k, t):
    e = 0.0
    for j in range(k + 1):
        pr = math.comb(k, j) * t ** j * (1 - t) ** (k - j)
        e += pr * (F(m + j, m + k, m + j) - F(m, m, m + j))
    return e
N_FR = 259452
def lb(x, t):
    gg = x.group_by("s").agg(k=pl.len(), m=pl.col("n_v10b").first())
    return 0.15 * sum(row_gain(m, k, t) for m, k in gg.select("m", "k").iter_rows()) / N_FR
for name, x in [("all 477", d), ("amp_only", d.filter(pl.col("sub") == "amp_only")),
                ("contentdrop (in place + suffix)", d.filter(pl.col("sub").str.starts_with("contentdrop"))),
                ("noise_to_noise", d.filter(pl.col("sub") == "noise_to_noise"))]:
    be = next((t / 100 for t in range(30, 100) if lb(x, t / 100) > 0), None)
    print(f"{name}: n {x.height}; LB at t=0.40 {lb(x,0.40):+.7f}, 0.60 {lb(x,0.60):+.7f}, 0.77 {lb(x,0.77):+.7f}, 0.79 {lb(x,0.79):+.7f}, "
          f"0.85 {lb(x,0.85):+.7f}, 0.93 {lb(x,0.93):+.7f}, 0.99 {lb(x,0.99):+.7f}; break-even {be}")
# safer subset: pure '&' <-> '+'/'et' swaps and compagnie<->cie abbreviations, nothing else changed
ABBR = {("&", "+"), ("&", "et"), ("compagnie", "cie"), ("& cie", "+ compagnie"), ("& cie", "et compagnie"), ("& cie", "+")}
safe = d.filter(pl.struct("ms", "mq").map_elements(lambda r: (r["ms"], r["mq"]) in ABBR, return_dtype=pl.Boolean))
print("safer subset n", safe.height, "S1 rows", safe["s"].n_unique(), "LB at t=0.97", f"{lb(safe,0.97):+.7f}", "t=0.99", f"{lb(safe,0.99):+.7f}")
safe.select("s", "q", "p2", "sub", "ms", "mq", "n_v10b", "sn", "qn", "sa", "qa").write_parquet(os.path.join(V, "fr-gathik-rest_fr_noise_in_safe.parquet"))
print(safe.select("sn", "qn").head(12))
