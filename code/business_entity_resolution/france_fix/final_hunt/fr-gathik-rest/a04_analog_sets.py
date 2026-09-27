"""US/India labelled analog for the same classes (noise-word in, typo/garble in), then build the France add-sets and
expected LB change."""
import os, sys
from collections import Counter
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from common import WORK, log
from ops import words, undot_legal, LEGAL, NOISE, DESC, LEGAL_CANON, _match

T = r"C:/ber_scratch/final/fr-gathik-rest"
NOISEX = NOISE | {"&", "+", "et", "and", "compagnie", "cie", "fils", "freres", "associes", "service", "services", "groupe", "partners"}

# ---------------- US/India analog
a = pl.read_parquet(r"C:/ber_scratch/gap/train_analog.parquet")
tr = pl.scan_parquet(os.path.join(WORK, "train_s1.parquet")).select("business_name").collect()["business_name"].to_list()
voc = Counter()
for n in tr:
    for w in set(words(undot_legal(n or ""))):
        voc[w] += 1
del tr


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


def klass(r):
    o = r["ops"]
    if not any(k in o for k in ("n_swap:", "n_add:", "n_drop:")):
        return "no_word_change"
    if "n_domain" in o or "n_squash" in o or " " not in (r["qn"] or "").strip():
        return "handle_single"
    ms, mq = wdiff(r["qn"], r["sn"])
    if not mq:
        return "drop_only"
    if any(w in DESC for w in mq):
        return "desc_in"
    if all(w in NOISEX for w in mq):
        return "noise_in"
    real = [w for w in mq if voc.get(w, 0) >= 5]
    return "realword_in" if real else "garble_in"


a = a.with_columns(k=pl.Series([klass(r) for r in a.iter_rows(named=True)]),
                   bad=pl.col("ops").str.contains(r"n_legal_change|n_legal_add|a_num_up"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
log("US/India analog by class (all / no legal-change-add, no num-up, pg>=0.9):\n" + str(
    a.group_by("k").agg(n=pl.len(), prec=pl.col("y").mean(),
                        n_f=(~pl.col("bad") & (pl.col("pg") >= 0.9)).sum(),
                        prec_f=pl.col("y").filter(~pl.col("bad") & (pl.col("pg") >= 0.9)).mean(),
                        n_f8=(~pl.col("bad") & (pl.col("pg") >= 0.8)).sum(),
                        prec_f8=pl.col("y").filter(~pl.col("bad") & (pl.col("pg") >= 0.8)).mean()).sort("n", descending=True)))
a.select("s", "q", "y", "pg", "k", "bad", "country").write_parquet(os.path.join(T, "analog_cls.parquet"))
