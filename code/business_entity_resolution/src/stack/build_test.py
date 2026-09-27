"""Test feature table for the countries with training labels (US, India) on the union of both pipelines' pairs
-> OUT/te.parquet.
Checks that the baseline p reproduces the blend_second.py probabilities in BLEND (before the list-mover overrides)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *
from common import log

G = os.path.join(WORK, "gathik", "v9")
s1 = s1_info("test")
S = s1.select("s")


def rd(path, cols, ren):
    """Rows of a test score file in the countries with training labels: q, s and the columns `cols`, renamed by `ren` and cast to float32."""
    return i64(pl.read_parquet(path, columns=["q", "s"] + cols)).join(S, on="s").rename(ren).with_columns(
        pl.col(list(ren.values())).cast(pl.Float32))


a = rd(os.path.join(WORK, "ce_b2", "test_scores_ce_smallfolds.parquet"), ["p2"], {"p2": "a"})
b = rd(os.path.join(WORK, "ce_b2", "test_scores_ce_bgefolds.parquet"), ["p2"], {"p2": "b"})
c = rd(os.path.join(WORK, "ce_b2", "test_scores_ce_e5lfolds.parquet"), ["p2"], {"p2": "c"})
og = rd(os.path.join(WORK, "test_scores_full_cons_test_b2.parquet"), ["p1", "p2"], {"p1": "op1", "p2": "op2"})
log(f"ours small {a.height} bge {b.height} e5l {c.height} gbdt {og.height}")
ours = a.join(b, on=["q", "s"], how="full", coalesce=True).join(c, on=["q", "s"], how="left").join(og, on=["q", "s"], how="left")
log(f"ours pairs {ours.height}; op2 null {ours['op2'].null_count()}")
del a, b, c, og
pg = rd(os.path.join(G, "ce_x_test_scores_ce_bgef0bgef1bgef2.parquet"), ["p2"], {"p2": "pg"})
gg = rd(os.path.join(G, "test_scores_full_xgb_cons.parquet"), ["p1", "p2"], {"p1": "gp1", "p2": "gp2"})
gat = pg.join(gg, on=["q", "s"], how="left")
log(f"gathik pairs {gat.height}; gp2 null {gat['gp2'].null_count()}")
del pg, gg
d = ours.join(gat, on=["q", "s"], how="full", coalesce=True)
del ours, gat
d = blend_p(d)
cu = i64(pl.read_parquet(os.path.join(BLEND, "cand_union.parquet"))).join(S, on="s")
log(f"union pairs {d.height}; cand_union US/India {cu.height}; in both {d.join(cu, on=['q','s']).height}")
ref = i64(pl.read_parquet(os.path.join(BLEND, "test_scores_usi_blend_w04_movers.parquet"))).rename({"p2": "pref"})
mv = pl.concat([pl.read_parquet(os.path.join(MOVERS, "tier12_removed.parquet"), columns=["q", "s"]),
                pl.read_parquet(os.path.join(MOVERS, "tier12_restored.parquet"), columns=["q", "s"])]).with_columns(mv=pl.lit(1))
chk = d.select("q", "s", "p").join(ref, on=["q", "s"], how="full", coalesce=True).join(mv, on=["q", "s"], how="left")
chk = chk.filter(pl.col("mv").is_null())
diff = chk.filter(((pl.col("p") - pl.col("pref")).abs() > 1e-5) | (pl.col("p").is_null() != pl.col("pref").is_null()))
log(f"blend reference pairs {ref.height}; non-mover pairs differing from this p: {diff.height}")
print(diff.head(5))
rec = rec_info("test")
d = add_feats(d, s1, rec)
os.makedirs(OUT, exist_ok=True)
d.write_parquet(os.path.join(OUT, "te.parquet"))
log(f"test feature rows {d.height}")
