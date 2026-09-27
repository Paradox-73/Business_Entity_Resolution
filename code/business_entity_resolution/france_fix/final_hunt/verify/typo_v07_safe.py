"""Safer subset of fr_typo_in + expected LB for full set and subsets (same per-row F0.5 model as the proposer, re-implemented)."""
import math, sys
import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(250)
V = r"C:/ber_scratch/final/verify"
x = pl.read_parquet(r"C:/ber_scratch/final/fr-gathik-rest/fr_typo_in.parquet")
m = pl.read_parquet(f"{V}/typo_mix.parquet").filter(pl.col("grp").str.starts_with("fr_set")).select("s", "q", "a", "b", "rec", "first")
t = pl.read_parquet(f"{V}/typo_twins.parquet")
d = x.join(m, on=["s", "q"], how="left").join(t, on=["s", "q"], how="left").with_columns(
    amiss=pl.col("opsF").str.contains("a_missing"))
print("multi-word garbles (not single-word):")
print(d.filter(pl.col("rec").is_null() & (pl.col("cls2") != "D_drop_stopword_only")).select("sn", "qn", "w_s", "w_q"))
print("records with a number not sharing the S1 house number:")
print(d.filter(~pl.col("num") & pl.col("qa").is_not_null() & pl.col("qa").str.contains(r"\d")).select("sn", "qn", "sa", "qa"))
good_letters = pl.col("first").fill_null(False) & (pl.col("rec").fill_null(0) >= 0.6)
multi_ok = pl.col("rec").is_null() & (pl.col("cls2") != "D_drop_stopword_only") & ~pl.col("w_q").str.contains(r"council|jaxonyx")
safe = d.filter((good_letters | (pl.col("cls2") == "D_drop_stopword_only") | multi_ok) & ~pl.col("amiss"))
rest = d.join(safe.select("s", "q"), on=["s", "q"], how="anti")
print("safe", safe.height, "rows", safe["s"].n_unique(), "; dropped", rest.height, dict(rest.group_by("cls2").len().iter_rows()))
N_FR = 259452


def F(tp, np_, nt):
    if nt == 0:
        return 1.0 if np_ == 0 else 0.0
    if tp == 0:
        return 0.0
    p, r = tp / np_, tp / nt
    return 1.25 * p * r / (0.25 * p + r)


def row_gain(mm, k, tt):
    return sum(math.comb(k, j) * tt ** j * (1 - tt) ** (k - j) * (F(mm + j, mm + k, mm + j) - F(mm, mm, mm + j)) for j in range(k + 1))


def lb(z, tt):
    g = z.group_by("s").agg(k=pl.len(), m=pl.col("n_v10b").first())
    return 0.15 * sum(row_gain(mm, k, tt) for mm, k in g.select("m", "k").iter_rows()) / N_FR


for name, z in [("full", d), ("safe", safe), ("dropped", rest)]:
    be = next((tt / 100 for tt in range(40, 100) if lb(z, tt / 100) > 0), None)
    print(name, z.height, " ".join(f"t={tt}:{lb(z, tt):+.7f}" for tt in (0.68, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99)), "break-even", be)
safe.select("s", "q", "p2", "cls2", "a", "b", "rec", "first", "sn", "qn", "sa", "qa", "n_v10b").write_parquet(
    f"{V}/fr-gathik-rest_fr_typo_in_safe.parquet")
print(safe.select("s", "q").dtypes)
