import os, sys, polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
from common import WORK  # noqa: E402
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../artifacts/census"))
from common import id_to_int
from ops import ops
W = f"{WORK}/"
OUT = f"{SCRATCH}/final2/fr-strong-veto/"
TMP = f"{SCRATCH}/"
i64 = lambda d: d.with_columns(pl.col("q").cast(pl.Int64), pl.col("s").cast(pl.Int64))
j = pl.read_parquet(OUT+"fr_v10c_scored.parquet")
nv = j.group_by("s").agg(n_v=pl.len())
j = j.join(nv, on="s")
C = j.filter(pl.col("occ") & pl.col("gcc") & (pl.col("ob") < 0.5) & (pl.col("gb") < 0.5)).with_columns(grp=pl.lit("cand"))
K = j.filter(pl.col("occ") & pl.col("gcc") & (pl.col("ob") > 0.9) & (pl.col("gb") > 0.9)).sample(15000, seed=1).with_columns(grp=pl.lit("keep_cc"))
M = j.filter(pl.col("occ") & pl.col("gcc") & (pl.col("ob").is_between(0.5, 0.9)) & (pl.col("gb").is_between(0.5,0.9))).with_columns(grp=pl.lit("mid_cc"))
c = pl.concat([C, K, M])
vetos = pl.concat([pl.read_parquet(TMP+"frfix/namechg/veto_set.parquet", columns=["q","s"]), pl.read_parquet(W+"frfix2/fp_veto_set.parquet", columns=["q","s"]),
                   pl.read_parquet(W+"frfix3/moreveto_stem_set.parquet", columns=["q","s"]), pl.read_parquet(W+"frfix3/moreveto_stem2_set.parquet", columns=["q","s"])]).pipe(i64).unique()
print("candidates already in veto sets:", C.join(vetos, on=["q","s"], how="semi").height)
s1 = pl.scan_parquet(W+"test_s1.parquet").filter(pl.col("country")=="France").select(s=id_to_int("entity_id"), sn="business_name", sa="business_address").collect()
rec = pl.concat([pl.scan_parquet(W+f"test_s{k}.parquet").select("entity_id","business_name","business_address") for k in (2,3)]).select(
    q=id_to_int("entity_id"), qn="business_name", qa="business_address").join(c.select("q").unique().lazy(), on="q").collect()
c = c.join(s1, on="s").join(rec, on="q")
c = c.with_columns(ops=pl.Series([";".join(sorted(ops(r["qn"] or "", r["qa"] or "", r["sn"] or "", r["sa"] or "", "France"))) for r in c.iter_rows(named=True)], dtype=pl.Utf8),
                   low=(pl.col("qn") == pl.col("qn").str.to_lowercase()) & pl.col("qn").str.contains("[a-z]"))
c = c.with_columns(nm=pl.col("ops").str.extract_all(r"n_[a-z_]+(?::[a-z]+)?").list.join("+"), num=pl.col("ops").str.extract(r"(a_num[a-z0-9_]*)"),
                   amiss=pl.col("ops").str.contains("a_missing"))
# twins: number of France S1 rows with same normalized S1 name
norm = lambda e: e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}","").str.to_lowercase().str.replace_all(r"[^a-z0-9]+"," ").str.strip_chars()
tw = s1.with_columns(k=norm(pl.col("sn"))).group_by("k").agg(ntw=pl.len())
c = c.with_columns(k=norm(pl.col("sn"))).join(tw, on="k", how="left").drop("k")
c.write_parquet(OUT+"fr_prof.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
print(c.group_by("grp").agg(n=pl.len(), low=pl.col("low").mean(), same_name=(pl.col("nm")=="").mean(), amiss=pl.col("amiss").mean(), num_up=pl.col("num").str.contains("up").mean(),
      num_dn=pl.col("num").str.contains("down").mean(), num_miss=(pl.col("num")=="a_num_missing").mean(), twin=(pl.col("ntw")>1).mean(), nv1=(pl.col("n_v")==1).mean(), nv=pl.col("n_v").mean(), og=pl.col("og").mean(), gg=pl.col("gg").mean()))
for g in ["cand"]:
    x = c.filter(pl.col("grp")==g)
    print(x.group_by("nm").agg(n=pl.len(), low=pl.col("low").mean(), up=pl.col("num").str.contains("up").mean()).sort("n", descending=True).head(30))
