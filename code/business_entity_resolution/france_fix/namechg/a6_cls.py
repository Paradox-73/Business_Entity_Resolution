"""Word classes from France keep rates; class composition of accepted / restored (v7m) / v7i / v7g_num changes."""
import sys
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
FR = "C:/ber_scratch/france"
OUT = "C:/ber_scratch/frfix/namechg"
kw = pl.read_parquet(f"{OUT}/fr_word_keep.parquet")
df = pl.read_parquet(f"{OUT}/s1_df.parquet")
kw = kw.join(df.rename({"w": "a"}), on="a", how="left").with_columns(pl.col("df_s1").fill_null(0))
GRP = {"groupe", "developpement", "france"}


def wclass(r):
    a, n, k, dfs = r["a"], r["n"], r["keep"], r["df_s1"]
    if a in GRP:
        return "G"
    if n < 40 or k is None or k != k:
        return "rare"
    if k < 0.08:
        return "X"
    if k >= 0.7:
        return "N"
    if 0.25 <= k < 0.5 and dfs >= 100:
        return "D"
    return "M"      # city names etc.


kw = kw.with_columns(cls=pl.Series([wclass(r) for r in kw.iter_rows(named=True)]))
print(kw.group_by("cls").agg(nw=pl.len(), n=pl.col("n").sum(), words=pl.col("a").head(12)).sort("n", descending=True))
kw.select("a", "cls", "keep", "n", "df_s1").write_parquet(f"{OUT}/word_class.parquet")
wc = dict(zip(kw["a"].to_list(), kw["cls"].to_list()))
ORDER = {"X": 0, "D": 1, "G": 2, "M": 3, "rare": 4, "N": 5}

c = pl.read_parquet(f"{OUT}/fr_chg.parquet", columns=["q", "s", "pat", "added", "dropped", "samestreet", "twin_samenum"])


def pcls(added, pat):
    nk = pat.split("|")[0]
    if not added:
        return "drop" if nk == "dropped" else nk
    cs = [wc.get(w, "rare") for w in added]
    return min(cs, key=lambda x: ORDER[x])


c = c.with_columns(cls=pl.Series([pcls(a, p) for a, p in zip(c["added"].to_list(), c["pat"].to_list())]))
c.select("q", "s", "cls").write_parquet(f"{OUT}/pair_class.parquet")
a = pl.read_parquet(f"{OUT}/fr_all.parquet").join(c.select("q", "cls", "samestreet"), on="q", how="left").with_columns(
    pl.col("cls").fill_null("samename"), num=pl.col("pat").str.split("|").list.last())
a.write_parquet(f"{OUT}/fr_all_cls.parquet")
x = a.filter(pl.col("num") == "nsame")
print("same number, by class: n, accepted, restored (v7m-v7ens), mean p3/g of accepted and restored")
print(x.group_by("cls").agg(n=pl.len(), acc=pl.col("acc").sum(), rest=(pl.col("in7m") & ~pl.col("acc")).sum(),
                            p3_acc=pl.col("p3").filter(pl.col("acc")).mean(), p3_rest=pl.col("p3").filter(pl.col("in7m") & ~pl.col("acc")).mean(),
                            g_rest=pl.col("p2g").filter(pl.col("in7m") & ~pl.col("acc")).mean()).sort("n", descending=True))
R = a.filter(pl.col("in7m") & ~pl.col("acc"))
print("restored total", R.height, R.group_by("cls", "num").len().sort("len", descending=True).head(12).to_dicts())
E = pl.read_parquet(f"{FR}/pairs_v7ens.parquet")
for nm in ("v7i", "v7g_num", "v7j"):
    V = pl.read_parquet(f"{FR}/pairs_{nm}.parquet")
    add = V.join(E, on=["q", "s"], how="anti").join(a, on=["q", "s"], how="left")
    rem = E.join(V, on=["q", "s"], how="anti").join(a, on=["q", "s"], how="left")
    print(nm, "added", add.height, add.group_by("cls", "num").len().sort("len", descending=True).head(10).to_dicts())
    print(nm, "removed", rem.height, rem.group_by("cls", "num").len().sort("len", descending=True).head(10).to_dicts())
