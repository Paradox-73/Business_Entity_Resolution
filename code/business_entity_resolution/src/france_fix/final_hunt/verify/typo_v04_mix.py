"""Mixture estimate of the decoy share from letter recall rec(a,b) = share of the S1 word a's letters found in record word b.
True garbles: analog y=1 garble_in (pg>=0.9, no legal/num-up). Decoy-like: France clean single-word swaps (X_desc_in, R_realword_in),
where b is the swapped real word B (a garble of B can only lose letters of B, so rec(a, garble(B)) is at most about rec(a, B))."""
import os, sys, math
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
import polars as pl
from ops import words, undot_legal, LEGAL, LEGAL_CANON, _match, DESC

pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(25)


def wdiff(qn, sn):
    qw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(qn or ""))]
    sw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(sn or ""))]
    qx = [w for w in qw if w not in LEGAL]
    sx = [w for w in sw if w not in LEGAL]
    used = [False] * len(qx)
    miss_s = []
    for x in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == x), None)
        if j is None:
            j = next((j for j, b in enumerate(qx) if not used[j] and _match(x, b)), None)
        if j is None:
            miss_s.append(x)
        else:
            used[j] = True
    return miss_s, [qx[j] for j in range(len(qx)) if not used[j]]


def rec(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum(min(v, cb[k]) for k, v in ca.items()) / max(len(a), 1)


def one(sn, qn):
    ms, mq = wdiff(qn, sn)
    if len(ms) != 1 or len(mq) != 1:
        return None
    return ms[0], mq[0]


rows = []
an = pl.read_parquet(rf"{SCRATCH}/gap/train_analog.parquet")
k = pl.read_parquet(rf"{SCRATCH}/final/fr-gathik-rest/analog_cls.parquet")
an = an.join(k.select("s", "q", "k", "bad"), on=["s", "q"]).filter((pl.col("k") == "garble_in") & ~pl.col("bad") & (pl.col("pg") >= 0.9))
for r in an.iter_rows(named=True):
    x = one(r["sn"], r["qn"])
    if x:
        rows.append(dict(grp=f"analog_garble_y{int(r['y'])}", a=x[0], b=x[1]))
rest = pl.read_parquet(rf"{SCRATCH}/final/fr-gathik-rest/rest_cls2.parquet")
for r in rest.filter(pl.col("cls2").is_in(["X_desc_in", "R_realword_in"])).iter_rows(named=True):
    x = one(r["sn"], r["qn"])
    if x:
        rows.append(dict(grp="fr_clean_swap_" + r["cls2"][:1], a=x[0], b=x[1]))
fr = pl.read_parquet(rf"{SCRATCH}/final/fr-gathik-rest/fr_typo_in.parquet")
for r in fr.iter_rows(named=True):
    x = one(r["sn"], r["qn"])
    if x:
        rows.append(dict(grp="fr_set_" + ("aDESC" if x[0] in DESC else "aOTHER"), a=x[0], b=x[1], s=r["s"], q=r["q"]))
d = pl.DataFrame(rows, infer_schema_length=None).with_columns(
    rec=pl.struct("a", "b").map_elements(lambda z: rec(z["a"], z["b"]), return_dtype=pl.Float64),
    first=pl.col("a").str.slice(0, 1) == pl.col("b").str.slice(0, 1))
print(d.group_by("grp").agg(n=pl.len(), first=pl.col("first").mean(), rec=pl.col("rec").mean(), lt05=(pl.col("rec") < 0.5).mean(),
                             lt06=(pl.col("rec") < 0.6).mean(), ge07=(pl.col("rec") >= 0.7).mean(),
                             first_lt05=((pl.col("rec") < 0.5) | ~pl.col("first")).mean()).sort("grp"))
# mixture fit on bin rec<0.5 (or first letter differs): F = (1-pi) T + pi D
T = d.filter(pl.col("grp") == "analog_garble_y1")
D = d.filter(pl.col("grp").str.starts_with("fr_clean_swap"))
for name, F in [("all", d.filter(pl.col("grp").str.starts_with("fr_set"))), ("aDESC", d.filter(pl.col("grp") == "fr_set_aDESC")),
                ("aOTHER", d.filter(pl.col("grp") == "fr_set_aOTHER"))]:
    for thr in (0.5, 0.6):
        f = lambda z: ((z["rec"] < thr) | ~z["first"]).mean()
        t, dd, ff, n = f(T), f(D), f(F), F.height
        pi = (ff - t) / (dd - t)
        ff_hi = ff + 1.645 * math.sqrt(ff * (1 - ff) / n)
        pi_hi = (ff_hi - t) / (dd - t)
        print(f"{name} n={n} thr={thr}: true-garble share below {t:.3f}, clean-swap-decoy share below {dd:.3f}, France set {ff:.3f} "
              f"-> decoy share {pi:.3f} (one-sided 95% upper {pi_hi:.3f})")
print(D.select("grp", "a", "b", "rec").sample(20, seed=1))
print(d.filter(pl.col("grp") == "fr_set_aDESC").sort("rec").select("a", "b", "rec", "first").head(60))
d.write_parquet(rf"{SCRATCH}/final/verify/typo_mix.parquet")
