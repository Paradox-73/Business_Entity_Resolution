import sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from common import id_to_int
W = "E:/Projects/Amazon ML Challenge/work/"
OUT = "C:/ber_scratch/final2/fr-strong-veto/"
c = pl.read_parquet(OUT+"fr_prof.parquet")
def cls(e):
    return (pl.when(e.str.contains("n_swap:noise|n_swap:desc") | (e.str.contains("n_drop") & e.str.contains("n_add:noise"))).then(pl.lit("drop/swap+noise"))
            .when(e.str.contains("n_legal")).then(pl.lit("legal")).when(e=="").then(pl.lit("same_name")).when(e.str.contains("n_add:noise") & ~e.str.contains("n_swap|n_drop")).then(pl.lit("add_noise_only")).otherwise(pl.lit("other")))
c = c.with_columns(k=cls(pl.col("nm")))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
print(c.group_by("grp").agg(n=pl.len(), amiss=pl.col("amiss").mean(), num_up=pl.col("num").str.contains("up").mean(), num_miss=(pl.col("num")=="a_num_missing").mean(), twin=(pl.col("ntw")>1).mean(), nv1=(pl.col("n_v")==1).mean()))
print(c.group_by("grp","k").len().pivot(on="grp", index="k", values="len"))
# noise tokens
n = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase()
toks = ["developpement","groupe","cie","fils","associes","france","ets","et "]
print(c.with_columns(qq=n(pl.col("qn"))).group_by("grp").agg([pl.col("qq").str.contains(t).mean().alias(t) for t in toks]))
# restrict to drop/swap+noise: score distribution in keep vs cand among all v10c cc pairs in this class
j = c.filter(pl.col("k")=="drop/swap+noise")
print(j.group_by("grp").agg(n=pl.len(), ob=pl.col("ob").mean(), gb=pl.col("gb").mean()))
# competing S1: best other S1 per record in ob / gb (France)
frs = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id")).collect()["s"]
cq = c.filter(pl.col("grp")=="cand").select("q","s")
ob = pl.scan_parquet(W+"ce_b2/test_scores_ce_bgefolds.parquet").filter(pl.col("q").is_in(cq["q"])).select("q","s","p2").collect()
gb = pl.scan_parquet(W+"gathik/v9/ce_x_test_scores_ce_bgef0bgef1bgef2.parquet").filter(pl.col("q").is_in(cq["q"])).select("q","s","p2").collect()
bo = ob.join(cq, on=["q","s"], how="anti").group_by("q").agg(ob_other=pl.col("p2").max())
bg = gb.join(cq, on=["q","s"], how="anti").group_by("q").agg(gb_other=pl.col("p2").max())
x = c.filter(pl.col("grp")=="cand").join(bo, on="q", how="left").join(bg, on="q", how="left")
x = x.with_columns(alt=((pl.col("ob_other")>=0.5)&(pl.col("gb_other")>=0.5)).fill_null(False), alt_any=((pl.col("ob_other")>=0.5)|(pl.col("gb_other")>=0.5)).fill_null(False))
print(x.group_by("k").agg(n=pl.len(), alt=pl.col("alt").mean(), alt_any=pl.col("alt_any").mean(), obo=pl.col("ob_other").mean()))
x.write_parquet(OUT+"fr_cand_alt.parquet")
