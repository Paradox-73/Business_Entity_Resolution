"""Handle-form records (@name, #name, namecom): does the handle spell the S1 name's leading words?"""
import sys, re, unicodedata
import os
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(50)
OUT = f"{SCRATCH}/frfix3/moreveto"
g = pl.read_parquet(f"{OUT}/grp_table2.parquet").filter(pl.col("handle") & (pl.col("pop") | pl.col("dv")))
def norm(x):
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
def core(q):
    x = q.strip()
    x = re.sub(r"^[@#]+", "", x)
    x = re.sub(r"(?i)\.?(com|fr|net|org)$", "", x)
    return re.sub(r"[^a-z0-9]", "", norm(x))
LEG = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "selarl", "scp", "gie", "earl"}
def check(q, s):
    c = core(q)
    ws = [re.sub(r"[^a-z0-9]", "", w) for w in norm(s).replace("&", " ").split()]
    ws = [w for w in ws if w]
    # accented letters may be dropped entirely ("medical" -> "mdical"): compare with those letters removed too
    ws2 = [re.sub(r"[^a-z0-9]", "", w) for w in re.sub(r"[^\x00-\x7f]", "", s).lower().replace("&", " ").split()]
    ws2 = [w for w in ws2 if w]
    for W in (ws, ws2, [w for w in ws if w not in LEG], [w for w in ws2 if w not in LEG]):
        for k in range(1, len(W) + 1):
            j = "".join(W[:k])
            if j == c:
                return "prefix%d" % min(k, 3)
        if c and c in "".join(W):
            return "substr"
    return "mismatch"
g = g.with_columns(hk=pl.Series([check(q, s) for q, s in zip(g["qn"].to_list(), g["sn"].to_list())]))
print(g.group_by("hk", "dv").agg(n=pl.len(), cur=pl.col("cur").sum()).sort("n", descending=True))
m = g.filter((pl.col("hk") == "mismatch") & pl.col("cur"))
print(m.select("sn", "qn", "npat", "p3").sample(min(40, m.height), seed=2))
g.select("q", "s", "hk", "cur", "dv").write_parquet(f"{OUT}/handle_chk.parquet")
