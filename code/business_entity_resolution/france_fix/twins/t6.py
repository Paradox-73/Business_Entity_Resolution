import polars as pl, sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
from france_cal import name_kind, toks
T = r"C:/ber_scratch/frfix/twins"
F = r"C:/ber_scratch/france"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(300)
s1 = pl.read_parquet(f"{T}/fr_s1k.parquet").filter(pl.col("snum").is_not_null() & (pl.col("sst") != ""))
recs = pl.read_parquet(f"{T}/fr_recsk.parquet")
b = pl.read_parquet(f"{T}/fr_pairs.parquet")
acc = pl.read_parquet(f"{F}/pairs_v7ens.parquet").with_columns(a=pl.lit(True))
grp = s1.group_by("snum", "sst", "scity").agg(ns=pl.len())
# record -> all S1 at the same address key
qa = (recs.filter(pl.col("qnum").is_not_null() & (pl.col("qst") != ""))
      .join(s1.select("s", "sn", "snum", "sst", "scity"), left_on=["qnum", "qst", "qcity"], right_on=["snum", "sst", "scity"]))
qa = qa.join(grp, left_on=["qnum", "qst", "qcity"], right_on=["snum", "sst", "scity"])
print("records with >=1 S1 at exact same address key:", qa["q"].n_unique(), " pairs:", qa.height)
qa = qa.filter(pl.col("ns") >= 2)
print("records whose address has >=2 S1 (twin address):", qa["q"].n_unique(), " pairs:", qa.height)
qa = qa.with_columns(nk=pl.Series([name_kind(s, q) for s, q in zip(qa["sn"].to_list(), qa["qn"].to_list())]))
qa = (qa.join(b.select("q", "s", "g", "p3", "pf", "pr"), on=["q", "s"], how="left")
        .join(acc, on=["q", "s"], how="left").with_columns(a=pl.col("a").fill_null(False), cand=pl.col("g").is_not_null()))
qa.select("q", "s", "qn", "sn", "qa", "nk", "ns", "g", "p3", "pf", "pr", "a", "cand").write_parquet(f"{T}/twin_addr.parquet")
print(qa.group_by("nk").agg(n=pl.len(), cand=pl.col("cand").mean(), acc=pl.col("a").mean()).sort("n", descending=True))
