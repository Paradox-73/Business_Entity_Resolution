"""The 2,441 France gathik-only pairs NOT added in v10b: enrich with v10b status, gathik scores, word-level diff, classes."""
import os, sys, re
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
import polars as pl
from common import WORK, id_to_int, log, read_tsv
from ops import ops, words, undot_legal, wclass, LEGAL, LEGAL_CANON, _match

R = os.path.dirname(WORK)
T = r"C:/ber_scratch/final/fr-gathik-rest"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))


def pairs(p):
    d = read_tsv(p)
    return (d.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
             .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))


free = pl.read_parquet(r"C:/ber_scratch/gap/fr_gathik_free.parquet")
added = pl.read_parquet(os.path.join(WORK, "out_v10b_fr", "france_recall_added.parquet"), columns=["s", "q"])
rest = free.join(added, on=["s", "q"], how="anti")
log(f"free {free.height}, added {added.height}, rest {rest.height}; rest q unique {rest['q'].n_unique()}, s unique {rest['s'].n_unique()}")

v10b = pairs(os.path.join(R, "submissions", "v10b", "matching_results.tsv"))
fr_s = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id").cast(pl.Int64))
v10b_fr = v10b.join(fr_s, on="s")
log(f"v10b France pairs {v10b_fr.height}")
rest = rest.join(v10b.select("q", s_v10b="s"), on="q", how="left")
log(f"rest records matched in v10b (to any S1): {rest.filter(pl.col('s_v10b').is_not_null()).height}")
cnt = v10b.group_by("s").len("n_v10b")
rest = rest.join(cnt, on="s", how="left").with_columns(pl.col("n_v10b").fill_null(0))

# veto sets
vs = [pl.read_parquet(r"C:/ber_scratch/frfix/namechg/veto_set.parquet", columns=["q", "s"]),
      pl.read_parquet(os.path.join(WORK, "frfix2", "fp_veto_set.parquet"), columns=["q", "s"]),
      pl.read_parquet(os.path.join(WORK, "frfix3", "moreveto_stem_set.parquet"), columns=["q", "s"]),
      pl.read_parquet(os.path.join(WORK, "frfix3", "moreveto_stem2_set.parquet"), columns=["q", "s"])]
veto = pl.concat([i64(v) for v in vs]).unique().with_columns(veto=pl.lit(True))
rest = rest.join(veto, on=["q", "s"], how="left").with_columns(pl.col("veto").fill_null(False))
log(f"rest pairs in a veto set: {rest['veto'].sum()}")

# gathik GBDT-only score
qs = rest.select("q").unique()
gx = i64(pl.scan_parquet(os.path.join(WORK, "gathik", "v9", "test_scores_full_xgb_cons.parquet")).join(qs.lazy(), on="q").collect())
rest = rest.join(gx.select("q", "s", pgx="p2"), on=["q", "s"], how="left")
# gathik's other candidates for the same record (count, best other)
go = gx.join(rest.select("q", "s"), on=["q", "s"], how="anti").sort("p2", descending=True).unique("q", keep="first").select("q", s_go="s", p_go="p2")
rest = rest.join(go, on="q", how="left")
# our lists: best S1 for the record (any), from our France combo scores
oc = i64(pl.scan_parquet(os.path.join(WORK, "frfix3", "test_scores_v9y_combo.parquet")).join(qs.lazy(), on="q").collect())
ob = oc.sort("p2", descending=True).unique("q", keep="first").select("q", s_our="s", p_our="p2")
rest = rest.join(ob, on="q", how="left")


# ops with country=France (postcodes skipped) and word-level diff
def wdiff(qn, sn):
    qw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(qn or ""))]
    sw = [LEGAL_CANON.get(w, w) for w in words(undot_legal(sn or ""))]
    qx = [w for w in qw if w not in LEGAL]
    sx = [w for w in sw if w not in LEGAL]
    used = [False] * len(qx)
    miss_s = []
    for a in sx:
        j = next((j for j, b in enumerate(qx) if not used[j] and b == a), None)
        if j is None:
            j = next((j for j, b in enumerate(qx) if not used[j] and _match(a, b)), None)
        if j is None:
            miss_s.append(a)
        else:
            used[j] = True
    miss_q = [qx[j] for j in range(len(qx)) if not used[j]]
    return " ".join(miss_s), " ".join(miss_q), len(sx)


rows = []
for r in rest.iter_rows(named=True):
    o = ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or "", "France")
    ms, mq, ns = wdiff(r["qn"], r["sn"])
    rows.append((";".join(sorted(o)), ms, mq, ns))
rest = rest.with_columns(opsF=pl.Series([x[0] for x in rows]), w_s=pl.Series([x[1] for x in rows]), w_q=pl.Series([x[2] for x in rows]),
                         n_sw=pl.Series([x[3] for x in rows]))
rest = rest.with_columns(
    nm=pl.col("opsF").str.extract_all(r"n_[a-z_]+(?::[a-z>]+)?").list.join("+"),
    num=pl.col("opsF").str.extract(r"(a_num_(?:up|down)[a-z0-9_]*)"),
    wordchg=pl.col("opsF").str.contains(r"n_swap:|n_add:|n_drop:"),
    legalbad=pl.col("opsF").str.contains(r"n_legal_change|n_legal_add"),
    numup=pl.col("opsF").str.contains(r"a_num_up"),
    lowp=pl.col("p2").fill_null(0) < 0.8,
    lowtest_ok=~pl.col("opsF").str.contains(r"n_domain|n_squash"),
)
rest.write_parquet(os.path.join(T, "rest.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
log(str(rest.group_by("wordchg", "legalbad", "numup", "lowp").agg(n=pl.len(), low=pl.col("low").mean(), p=pl.col("p2").mean(),
                                                                   s_has=(pl.col("n_v10b") > 0).mean()).sort("n", descending=True)))
wc = rest.filter(pl.col("wordchg"))
log(str(wc.with_columns(wk=pl.col("opsF").str.extract_all(r"n_(?:swap|add|drop):[a-z>]+").list.join("+")).group_by("wk").agg(
    n=pl.len(), low=pl.col("low").mean(), p=pl.col("p2").mean(), up=pl.col("numup").mean(),
    down=pl.col("num").str.contains("down").mean(), s_has=(pl.col("n_v10b") > 0).mean()).sort("n", descending=True).head(40)))
