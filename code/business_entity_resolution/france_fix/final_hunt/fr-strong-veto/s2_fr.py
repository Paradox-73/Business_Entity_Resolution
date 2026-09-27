import sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import id_to_int
W = "E:/Projects/Amazon ML Challenge/work/"
OUT = "C:/ber_scratch/final2/fr-strong-veto/"
v = pl.read_parquet(OUT+"v10c_fr_pairs.parquet")
frs = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(id_to_int("entity_id").alias("s")).collect()["s"]
def sc(f, cols):
    return pl.scan_parquet(f).filter(pl.col("s").is_in(frs)).select(["q","s"]+cols).collect()
# our close-call rows raw ce (3 folds)
ce = None
for k in range(3):
    d = sc(W+f"ce_b2/test_ce_bgef{k}.parquet", ["ce"]).rename({"ce":f"ce{k}"})
    ce = d if ce is None else ce.join(d, on=["q","s"], how="full", coalesce=True)
print(ce.describe())
ob = sc(W+"ce_b2/test_scores_ce_bgefolds.parquet", ["p2"]).rename({"p2":"ob"})
oe = sc(W+"ce_b2/test_scores_ce_e5lfolds.parquet", ["p2"]).rename({"p2":"oe"})
osm = sc(W+"ce_b2/test_scores_ce_smallfolds.parquet", ["p2"]).rename({"p2":"osm"})
og = sc(W+"test_scores_full_cons.parquet", ["p2"]).rename({"p2":"og"})
gr = sc(W+"gathik/v9/ce_x_test_rows.parquet", ["p2"]).rename({"p2":"gcc_p2"})
gb = sc(W+"gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet", ["p2"]).rename({"p2":"gb"})
gm = sc(W+"gathik/v9/ce_x_test_scores_v9_frmin.parquet", ["p2"]).rename({"p2":"gmin"})
gg = sc(W+"gathik/v9/test_scores_full_xgb_cons.parquet", ["p2"]).rename({"p2":"gg"})
j = v
for d in [ce, ob, oe, osm, og, gr, gb, gm, gg]:
    j = j.join(d, on=["q","s"], how="left")
j = j.with_columns(occ=pl.col("ce0").is_not_null(), gcc=pl.col("gcc_p2").is_not_null())
j.write_parquet(OUT+"fr_v10c_scored.parquet")
print(j.describe())
print(j.group_by("occ","gcc").len())
for t in [0.1,0.2,0.3,0.5]:
    a = j.filter(pl.col("occ") & pl.col("gcc") & (pl.col("ob")<t) & (pl.col("gb")<t))
    b = a.filter(pl.col("oe")<t)
    print(t, "both-bge-low", a.height, "+e5l low", b.height, "rows", a["s"].n_unique())
