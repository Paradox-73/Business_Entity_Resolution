# address keys for France S1 and records; S1 twin groups
import polars as pl, sys, re, unicodedata
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/france_fix/: shared France helpers
SCRATCH = os.environ.get("BER_SCRATCH", "C:/ber_scratch")  # scratch folder of the analysis runs
from fr_restore import street
T = rf"{SCRATCH}/frfix/twins"
REG = {"hauts de france", "nouvelle aquitaine", "pays de la loire", "nord", "loire atlantique", "gironde", "pas de calais", "france", "vendee", "maine et loire", "sarthe", "mayenne", "somme", "oise", "aisne", "charente", "charente maritime", "dordogne", "landes", "lot et garonne", "pyrenees atlantiques"}
def nrm(x):
    x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", x).strip()
def city(a):
    parts = [nrm(p) for p in (a or "").split(",")]
    c = [p for p in parts if p and not re.search(r"\d", p) and p not in REG]
    c = [re.sub(r"^st ", "saint ", p) for p in c]
    return c[-1] if c else ""
def keys(df, col, pre):
    st = [street(a) for a in df[col].to_list()]
    return df.with_columns(pl.Series(pre + "num", [x[0] for x in st], dtype=pl.Utf8), pl.Series(pre + "st", [x[1] for x in st]),
                           pl.Series(pre + "city", [city(a) for a in df[col].to_list()]))
s1 = keys(pl.read_parquet(f"{T}/fr_s1.parquet"), "sa", "s")
s1.write_parquet(f"{T}/fr_s1k.parquet")
recs = keys(pl.read_parquet(f"{T}/fr_recs.parquet"), "qa", "q")
recs.write_parquet(f"{T}/fr_recsk.parquet")
print(s1["scity"].value_counts().sort("count", descending=True).head(20))
print(recs["qcity"].value_counts().sort("count", descending=True).head(20))
g = s1.filter(pl.col("snum").is_not_null() & (pl.col("sst") != "")).group_by("snum", "sst", "scity").len()
print("S1 with address key:", g["len"].sum(), "; same-address group sizes:", g["len"].value_counts().sort("len").head(10))
