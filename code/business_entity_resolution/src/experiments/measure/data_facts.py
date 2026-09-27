"""Data facts of the methodology document (section 2.1): row counts, label structure, chains, the lowercase share of
records with and without a word change, and the house-number difference of true pairs and decoys.

  python data_facts.py        (from any folder; reads BER_WORK and the train ground truth, about 4 GB of RAM)

Reads WORK/{train,test}_s{1,2,3}.parquet (README step 1), the train ground truth and the out-of-fold stage-1 table
WORK/models/full_cons/oof.parquet (README step 4; decoys are compared with their best stage-1 candidate).
Writes $BER_SCRATCH/measure/data_facts.json (BER_SCRATCH defaults to C:/ber_scratch).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
import polars as pl  # noqa: E402
from common import WORK, id_to_int, read_truth  # noqa: E402

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
os.makedirs(SCR, exist_ok=True)
out = {}


def rd(split, k, cols):
    """Columns `cols` of the cleaned source file WORK/<split>_s<k>.parquet."""
    return pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"), columns=cols)


# ---- size: S1 rows and records per country
for split in ("train", "test"):
    s1 = rd(split, 1, ["entity_id", "country"])
    rec = pl.concat([rd(split, k, ["entity_id", "country"]) for k in (2, 3)])
    a = s1.group_by("country").len().rename({"len": "s1"}).join(rec.group_by("country").len().rename({"len": "rec"}), on="country")
    a = a.with_columns(per=pl.col("rec") / pl.col("s1")).sort("country")
    out[split] = {"s1": s1.height, "rec": rec.height, "by_country": a.rows()}
    print(split, s1.height, rec.height)
    print(a)

# ---- label structure (train)
tr = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id"))
s1 = rd("train", 1, ["entity_id", "country", "business_name", "name_core", "addr_nums"]).with_columns(s=id_to_int("entity_id"))
rec = pl.concat([rd("train", k, ["entity_id", "country", "business_name", "name_core", "addr_nums", "addr_missing",
                                 "name_nonlatin", "name_is_domain"]) for k in (2, 3)]).with_columns(q=id_to_int("entity_id"))
out["pairs"] = tr.height
out["distinct_q"] = tr["q"].n_unique()
out["s1_with_match"] = tr["s"].n_unique()
out["matched_share_of_records"] = out["distinct_q"] / rec.height
out["singleton_share"] = 1 - out["s1_with_match"] / s1.height
out["pairs_per_s1"] = tr.height / s1.height
j = tr.join(s1.select("s", sc="country"), on="s").join(rec.select("q", qc="country"), on="q")
out["same_country_share"] = (j["sc"] == j["qc"]).mean()
# chains: exact raw name shared with another S1 row of the same country; the same after cleaning to the core name
nc = s1.group_by("country", "business_name").len()
s1n = s1.join(nc, on=["country", "business_name"])
out["s1_shared_exact_name_share"] = (s1n["len"] >= 2).mean()
out["top_names"] = nc.sort("len", descending=True).head(3).rows()
nc2 = s1.group_by("country", "name_core").len()
s1c = s1.join(nc2, on=["country", "name_core"])
out["s1_shared_core_name_share"] = (s1c["len"] >= 2).mean()
print(json.dumps({k: v for k, v in out.items() if k not in ("train", "test")}, indent=1, default=str))

# ---- all-lowercase share of Latin, non-domain record names: decoys vs true records with / without a changed word
lat = rec.filter(~pl.col("name_nonlatin") & ~pl.col("name_is_domain") & pl.col("business_name").str.contains(r"[A-Za-z]"))
lat = lat.with_columns(low=(pl.col("business_name") == pl.col("business_name").str.to_lowercase()))
tq = tr.select("q", "s").unique("q")
lat = lat.join(tq, on="q", how="left")
dec = lat.filter(pl.col("s").is_null())
tru = lat.filter(pl.col("s").is_not_null()).join(s1.select("s", s_core="name_core", s_nums="addr_nums"), on="s")
tru = tru.with_columns(rw=pl.col("name_core").str.split(" "), sw=pl.col("s_core").str.split(" "))
tru = tru.with_columns(wchg=(pl.col("rw").list.set_difference("sw").list.len() > 0)
                       | (pl.col("sw").list.set_difference("rw").list.len() > 0))
lowres = {}
for c in ("US", "India"):
    d = dec.filter(pl.col("country") == c)
    t = tru.filter(pl.col("country") == c)
    lowres[c] = {"decoy_n": d.height, "decoy_low": d["low"].mean(), "true_n": t.height, "true_low": t["low"].mean(),
                 "true_wordchange_n": t.filter("wchg").height, "true_wordchange_low": t.filter("wchg")["low"].mean(),
                 "true_samewords_low": t.filter(~pl.col("wchg"))["low"].mean()}
out["lowercase"] = lowres
print(json.dumps(lowres, indent=1))


# ---- house-number difference: true pairs vs decoys' best stage-1 candidate (out-of-fold table)
def band(col):
    """Band of a house-number difference, as in the methodology table."""
    return (pl.when(pl.col(col).is_null()).then(pl.lit("no number"))
            .when(pl.col(col) == 0).then(pl.lit("0"))
            .when(pl.col(col) == 1).then(pl.lit("+1"))
            .when(pl.col(col).is_between(2, 5)).then(pl.lit("+2..5"))
            .when(pl.col(col).is_between(6, 20)).then(pl.lit("+6..20"))
            .when(pl.col(col).is_between(-20, -1)).then(pl.lit("-1..-20"))
            .otherwise(pl.lit("other")))


def first_num(c):
    """First house number of an address, as a float (null when there is none)."""
    return pl.col(c).list.first().cast(pl.Float64, strict=False)


oof = pl.read_parquet(os.path.join(WORK, "models", "full_cons", "oof.parquet"), columns=["q", "s", "p1"])
best = oof.sort("p1", descending=True).unique("q", keep="first")
recn = rec.select("q", "country", q_hn=first_num("addr_nums"))
s1h = s1.select("s", s_hn=first_num("addr_nums"))
dq = (best.join(tq, on="q", how="anti").join(recn, on="q").join(s1h, on="s")
      .with_columns(d=pl.col("q_hn") - pl.col("s_hn")).with_columns(b=band("d")))
tp = tr.join(recn, on="q").join(s1h, on="s").with_columns(d=pl.col("q_hn") - pl.col("s_hn")).with_columns(b=band("d"))
hn = {}
for c in ("US", "India"):
    x = dq.filter(pl.col("country") == c)
    y = tp.filter(pl.col("country") == c)
    hn[c] = {"decoys_n": x.height,
             "decoys": dict(x.group_by("b").len().with_columns(sh=pl.col("len") / x.height).select("b", "sh").rows()),
             "true_n": y.height,
             "true": dict(y.group_by("b").len().with_columns(sh=pl.col("len") / y.height).select("b", "sh").rows())}
out["house_number"] = hn
print(json.dumps(hn, indent=1))
json.dump(out, open(os.path.join(SCR, "data_facts.json"), "w"), indent=1, default=str)
print("wrote", os.path.join(SCR, "data_facts.json"))
