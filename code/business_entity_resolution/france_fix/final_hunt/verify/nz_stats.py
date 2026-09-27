import os, sys, math
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from common import WORK, id_to_int, read_tsv
from ops import ops, words, undot_legal, LEGAL, LEGAL_CANON, NOISE, DESC, STOPW, _match
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
R = os.path.dirname(WORK)
V = r"C:/ber_scratch/final/verify"
NOISEX = NOISE | {"&", "+", "et", "and", "compagnie", "cie", "fils", "freres", "associes", "service", "services", "groupe", "partners"}
AMP = {"&", "+", "et", "and"}

def diff(qn, sn):
    qw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(qn or ""))]
    sw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(sn or ""))]
    qx = [w for w in qw if w not in LEGAL]; sx = [w for w in sw if w not in LEGAL]
    used = [False] * len(qx); ms = []
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None:
            j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
        if j is None: ms.append(a)
        else: used[j] = True
    mq = [qx[j] for j in range(len(qx)) if not used[j]]
    # is a record-only noise word placed after the last legal token?
    li = max((i for i, w in enumerate(qw) if w in LEGAL), default=-1)
    suffix = li >= 0 and any(w in mq for w in qw[li + 1:])
    return ms, mq, suffix

def sub(ms, mq, suffix):
    if not mq: return "none_in"
    if any(w in DESC for w in mq) or not all(w in NOISEX for w in mq): return "not_noise_in"
    content_lost = [w for w in ms if w not in NOISEX and w not in STOPW]
    if not content_lost:
        return "amp_only" if all(w in AMP for w in ms + mq) else "noise_to_noise"
    return "contentdrop_suffix" if suffix else "contentdrop_inplace"

def lowf(n):
    return bool(n) and n == n.lower() and any(c.isalpha() for c in n)

def annotate(df, qn="qn", qa="qa", sn="sn", sa="sa", country=None):
    rows = []
    for r in df.select(qn, qa, sn, sa).iter_rows():
        o = ops(r[0] or "", r[1] or "", r[2] or "", r[3] or "", country)
        ms, mq, suf = diff(r[0], r[2])
        rows.append((sub(ms, mq, suf), " ".join(ms), " ".join(mq), lowf(r[0]),
                     any(k in o for k in ("n_domain", "n_squash")) or " " not in (r[0] or "").strip(),
                     any(k.startswith("a_num_up") for k in o), any(k.startswith("a_num_down") for k in o),
                     "n_legal_change" in o or "n_legal_add" in o, any(k.startswith(("n_swap:", "n_add:", "n_drop:")) for k in o),
                     any(k.startswith("n_") and k.split(":")[0] in ("n_swap", "n_add", "n_drop") and "desc" in k for k in o)))
    cols = ["sub", "ms", "mq", "lowr", "lowbad", "up", "down", "legbad", "wordchg", "desc_op"]
    return pl.concat([df, pl.DataFrame(rows, schema=cols, orient="row")], how="horizontal")

def summ(df, by):
    ok = ~pl.col("lowbad")
    return df.group_by(by).agg(n=pl.len(), n_lowok=ok.sum(), low=pl.col("lowr").filter(ok).sum(),
                               low_rate=pl.col("lowr").filter(ok).mean(), up=pl.col("up").sum(), down=pl.col("down").sum(),
                               legbad=pl.col("legbad").sum()).sort(by)

# ---------- 1. the proposed 477: raw names re-read from test parquet
d = pl.read_parquet(os.path.join(r"C:/ber_scratch/final/fr-gathik-rest", "fr_noise_in.parquet")).select("s", "q", "p2", "n_v10b", "low")
rec = pl.concat([pl.scan_parquet(os.path.join(WORK, f"test_s{k}.parquet")).select(q=id_to_int("entity_id").cast(pl.Int64), qn="business_name", qa="business_address")
                 .join(d.select("q").lazy(), on="q").collect() for k in (2, 3)])
s1 = pl.scan_parquet(os.path.join(WORK, "test_s1.parquet")).select(s=id_to_int("entity_id").cast(pl.Int64), sn="business_name", sa="business_address").join(d.select("s").unique().lazy(), on="s").collect()
d = d.join(rec, on="q").join(s1, on="s")
d = annotate(d, country="France")
print("proposer low flag == my raw low flag:", (d["low"] == d["lowr"]).all(), "; my low count", d["lowr"].sum())
print("== PROPOSED 477 by subclass"); print(summ(d, "sub"))
d.write_parquet(os.path.join(V, "nz_477.parquet"))

# ---------- 2. same population before the proposer's filters (rest_cls2 N_noise_in, any p/legal/num)
rc = pl.read_parquet(r"C:/ber_scratch/final/fr-gathik-rest/rest_cls2.parquet")
print("rest_cls2 columns:", rc.columns[:40])
rn = annotate(rc.select("s", "q", "qn", "qa", "sn", "sa", "p2", "cls2"), country="France")
print("== ALL 2,441 rest pairs by my subclass (no filters)"); print(summ(rn, "sub"))
print("== rest pairs, p2>=0.9, by my subclass"); print(summ(rn.filter(pl.col("p2") >= 0.9), "sub"))
rn.write_parquet(os.path.join(V, "nz_rest.parquet"))

# ---------- 3. the 4,825 already added in v10b (~99% true by LB): calibration of low flag for no-word-change true records
fg = pl.read_parquet(r"C:/ber_scratch/gap/fr_gathik_free.parquet")
add = pl.read_parquet(os.path.join(WORK, "out_v10b_fr", "france_recall_added.parquet"), columns=["s", "q"]).with_columns(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64))
fa = fg.join(add, on=["s", "q"])
print("4,825 v10b-added: lowercase share", fa.select(pl.col("low").mean()).item(), "n", fa.height)
