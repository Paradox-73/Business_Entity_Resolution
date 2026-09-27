import sys, polars as pl
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/france_fix/artifacts/census")
from common import id_to_int
from ops import ops
W = "E:/Projects/Amazon ML Challenge/work/"
OUT = "C:/ber_scratch/final2/fr-strong-veto/"
c = pl.concat([pl.read_parquet(OUT+"ho_cand_ab.parquet").with_columns(w=pl.lit("ab")), pl.read_parquet(OUT+"ho_cand_sm.parquet").with_columns(w=pl.lit("sm"))], how="diagonal")
s1 = pl.scan_parquet(W+"train_s1.parquet").select(s=id_to_int("entity_id"), sn="business_name", sa="business_address").join(c.select("s").unique().lazy(), on="s").collect()
rec = pl.concat([pl.scan_parquet(W+f"train_s{k}.parquet").select("entity_id","business_name","business_address") for k in (2,3)]).select(
    q=id_to_int("entity_id"), qn="business_name", qa="business_address").join(c.select("q").unique().lazy(), on="q").collect()
c = c.join(s1, on="s").join(rec, on="q")
c = c.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or "", r["country"]))) for r in c.iter_rows(named=True)], dtype=pl.Utf8),
                   low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
c = c.with_columns(nm=pl.col("ops").str.extract_all(r"n_[a-z_]+(?::[a-z]+)?").list.join("+"), num=pl.col("ops").str.extract(r"(a_num[a-z0-9_]*)"))
# simplified class
def cls(e):
    return (pl.when(e.str.contains("n_swap:noise|n_swap:desc") | (e.str.contains("n_drop") & e.str.contains("n_add:noise"))).then(pl.lit("drop/swap+noise"))
            .when(e.str.contains("n_legal")).then(pl.lit("legal")).when(e=="").then(pl.lit("same_name")).when(e.str.contains("n_add:noise") & ~e.str.contains("n_swap|n_drop")).then(pl.lit("add_noise_only")).otherwise(pl.lit("other")))
c = c.with_columns(k=cls(pl.col("nm")))
c.write_parquet(OUT+"ho_prof.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
for w in ["ab","sm"]:
    x = c.filter(pl.col("w")==w)
    for t in [0.1, 0.3, 0.5]:
        y = x.filter((pl.col("ob")<t)&(pl.col("gb")<t))
        print(w, t, y.group_by("k").agg(n=pl.len(), true=pl.col("label").mean(), up=pl.col("num").str.contains("up").mean(), low=pl.col("low").mean()).sort("k"))
x = c.filter((pl.col("w")=="ab")&(pl.col("ob")<0.3)&(pl.col("gb")<0.3))
print(x.select("country","sn","qn","sa","qa","nm","num","label",pl.col("ob").round(3),pl.col("gb").round(3)).sample(30, seed=2))
