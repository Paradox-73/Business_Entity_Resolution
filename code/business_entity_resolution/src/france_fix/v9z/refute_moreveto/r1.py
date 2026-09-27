import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl, numpy as np
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(60)
OUT = f"{SCRATCH}/frfix3/moreveto"
NC = f"{SCRATCH}/frfix/namechg"
g = pl.read_parquet(f"{OUT}/grp_table2.parquet")
print(g.columns, g.height)
x = g.filter(pl.col("grp").is_in(["swap_D_notveto", "add2+_D"]) & (pl.col("pop") | pl.col("cur")))
x = x.with_columns(stem=pl.struct("added", "dropped").map_elements(
    lambda r: any(p[:5] == q[:5] for p in r["added"] for q in r["dropped"] if len(p) >= 5 and len(q) >= 5), return_dtype=pl.Boolean))
KEEP_ADDED = {"compagnie", "etablissements", "etablissement"}
def abbrev(added, dropped):
    for a in added or []:
        if a in KEEP_ADDED: return True
        for d in dropped or []:
            sh, lo = (a, d) if len(a) <= len(d) else (d, a)
            if lo.startswith(sh) and len(sh) <= 4: return True
    return False
x = x.with_columns(abbr=pl.Series([abbrev(a, d) for a, d in zip(x["added"].to_list(), x["dropped"].to_list())], dtype=pl.Boolean))
S = x.filter(pl.col("stem") & ~pl.col("abbr"))
print("stem set: pop", S["pop"].sum(), "cur", S["cur"].sum(), "pop&cur", (S["pop"] & S["cur"]).sum(), "rows", S.height)
print(S.group_by("pop", "cur", "lowx", "acc", "dv", "ad", "fv").len())
print("lowercase rows:")
print(S.filter(pl.col("lowx")).select("q","s","sn","qn","npat","pat","cur","pop","acc","ad","fv","p2","p2g","p3","nc"))
print("pop not cur:")
print(S.filter(pl.col("pop") & ~pl.col("cur")).select("sn","qn","npat","cur","acc","ad","fv","p2","p3","lowx"))
print("all cur by dropped->added:")
print(S.filter(pl.col("cur")).group_by(pl.col("dropped").list.join(" ").alias("d"), pl.col("added").list.join(" ").alias("a"), "npat").agg(n=pl.len(), nlow=pl.col("lowx").sum()).sort("n", descending=True))
