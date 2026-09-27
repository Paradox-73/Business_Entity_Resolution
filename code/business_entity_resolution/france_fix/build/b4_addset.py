"""V2 add set: rejected (v7ens) best-candidate France pairs, swap/added kind, same first house number and same street,
added words of class N (noise) or G (groupe/developpement/france); G only where v7m restored it (g>=0.5, crossed 0.5)."""
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
RD = "C:/ber_scratch/frfix/refute_desc"
NC = "C:/ber_scratch/frfix/namechg"
FR = "C:/ber_scratch/france"
OUT = "C:/ber_scratch/frfix/build"
a = pl.read_parquet(f"{RD}/fr_low.parquet").join(pl.read_parquet(f"{NC}/fr_all_cls.parquet", columns=["q", "s", "samestreet", "p2"]), on=["q", "s"], how="left")
base = (~pl.col("acc") & ~pl.col("veto") & pl.col("kind").is_in(["swap", "added"]) & (pl.col("num") == "nsame") & pl.col("samestreet").fill_null(False))
addG = a.filter(base & (pl.col("cls") == "G") & pl.col("in7m"))
addN = a.filter(base & (pl.col("cls") == "N"))
add = pl.concat([addG, addN]).unique(["q", "s"])
Ld, Lt = 0.0338, 0.0008
for nm, x in (("G (v7m-restored)", addG), ("N", addN), ("ALL", add)):
    L = x["low"].mean(); print(f"{nm:18s} n={x.height:6d} lower={L:.4f} true~{(Ld-L)/(Ld-Lt):.2f} in7m={x['in7m'].sum()} mean p3 {x['p3'].mean():.3f} g {x['p2g'].mean():.3f} p2 {x['p2'].mean():.3f}")
print("S1 rows touched", add["s"].n_unique(), "; records", add["q"].n_unique())
t = pl.read_parquet(f"{FR}/fr_top.parquet", columns=["q", "s", "qn", "sn", "qa", "sa", "s_2", "p2_2"])
add = add.join(t, on=["q", "s"], how="left")
print("records whose 2nd candidate has p2>=0.5:", (add["p2_2"] >= 0.5).sum())
print(add.sample(25, seed=1).select("cls", "kind", "sn", "qn", "p3", "p2g"))
add.select("q", "s", "cls", "kind", "in7m", "p3", "p2g", "p2", "low").write_parquet(f"{OUT}/add_set_v2.parquet")
