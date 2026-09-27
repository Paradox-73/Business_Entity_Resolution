import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # src/
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../same-address"))
from salib import *
from common import WORK, id_to_int
from ops import words, STOPW, LEGAL, undot_legal
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
D = rf"{SCRATCH}/final/same-address/"
core = lambda n: [w for w in words(undot_legal(n or "")) if w not in LEGAL and w not in STOPW]
def ini(n):
    c = core(n); return "".join(w[0] for w in c) if len(c) >= 2 else ""
k = pl.read_parquet(D + "fr_all_keep.parquet", columns=["q", "s", "qn", "k", "nb", "v_s", "acr", "keep"]).filter(pl.col("keep") & pl.col("acr"))
k = k.filter(pl.col("v_s").is_null() | (pl.col("v_s") == pl.col("s"))).with_columns(grp=pl.when(pl.col("v_s").is_null()).then(pl.lit("unmatched")).otherwise(pl.lit("matched")))
s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country", "business_name", "business_address"]).filter(pl.col("country") == "France").select(
    s=id_to_int("entity_id").cast(pl.Int64), country="country", sn="business_name", sa="business_address")
s1 = s1.filter(pl.col("sa").is_not_null())
s1 = add_key(s1, "sa").filter(pl.col("k").is_in(k["k"].unique().implode()))
s1 = s1.with_columns(ini=pl.col("sn").map_elements(ini, return_dtype=pl.Utf8))
blk = s1.group_by("k").agg(inis=pl.col("ini"), ss=pl.col("s"))
k = k.join(blk, on="k", how="left")
def oneoff(tok, inis, ss, s):
    tok = tok.lower()
    n = 0
    for x, sx in zip(inis, ss):
        if sx == s or len(x) != len(tok): continue
        if sum(1 for a, b in zip(x, tok) if a != b) == 1: n += 1
    return n
toks = [re.sub(r"[^a-z]", "", "".join(core(q)[0:1])) if core(q) else "" for q in k["qn"].to_list()]
k = k.with_columns(tok=pl.Series(toks))
k = k.with_columns(n_oneoff=pl.Series([oneoff(t, i, ss, s) for t, i, ss, s in zip(k["tok"], k["inis"], k["ss"], k["s"])]))
k = k.with_columns(nbb=pl.col("nb").clip(1, 6), has1=pl.col("n_oneoff") > 0)
print(k.group_by("nbb", "grp").agg(n=pl.len(), oneoff_rate=pl.col("has1").mean()).sort("nbb", "grp"))
# expected one-off count among unmatched if they behaved like matched, per nb bucket
m = k.filter(pl.col("grp") == "matched").group_by("nbb").agg(r=pl.col("has1").mean())
u = k.filter(pl.col("grp") == "unmatched").join(m, on="nbb")
print("unmatched acronym records:", u.height, " observed with one-off neighbour:", u["has1"].sum(), " expected at matched rates:", round(u["r"].sum(), 1))
k.select("q", "s", "grp", "nb", "n_oneoff").write_parquet(rf"{SCRATCH}/final/verify/sa/acr_oneoff.parquet")
