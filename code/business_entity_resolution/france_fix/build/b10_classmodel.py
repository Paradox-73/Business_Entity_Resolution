"""Class-informed macro expected-F0.5 model: the lbcal logistic model (p3, g, pattern) with the lowercase-test class rates
fixed for descriptor-class (D/X) pairs and noise/groupe-class (N/G) same-address pairs; remaining parameters fitted to the
France LB numbers. Predicts V1 (descveto) and V2 (descveto + G/N additions). Also leave-v7m-out."""
import sys
sys.path.insert(0, "C:/ber_scratch/frfix/lbcal")
sys.path.insert(0, r"E:/Projects/Amazon ML Challenge/code/business_entity_resolution/src")
import numpy as np, polars as pl, time
from scipy.optimize import least_squares
import model as MD
from model import *
from common import WORK, id_to_int
NC = "C:/ber_scratch/frfix/namechg"; W = "E:/Projects/Amazon ML Challenge/work/frfix"
keys = b.select("q", "s").with_row_index("i")
pc = pl.read_parquet(f"{NC}/pair_class.parquet")
cls = keys.join(pc, on=["q", "s"], how="left").sort("i")["cls"].fill_null("none").to_numpy()
isveto = keys.join(pl.read_parquet(f"{NC}/veto_set.parquet", columns=["q", "s"]).with_columns(v=pl.lit(True)), on=["q", "s"], how="left").sort("i")["v"].fill_null(False).to_numpy()
fr = set(pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"].cast(pl.Utf8).to_list())
def mask_of(tsv):
    d = pl.read_csv(tsv, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}, quote_char=None)
    d = d.filter(pl.col("source1_entity_id").is_in(fr) & pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pr = d.with_columns(q=pl.col("matched_entity_ids").str.split(",")).explode("q").select(q=id_to_int("q"), s=id_to_int("source1_entity_id"))
    j = keys.join(pr.with_columns(m=pl.lit(True)), on=["q", "s"], how="left").sort("i")
    assert j["m"].sum() == pr.height, (j["m"].sum(), pr.height)
    return j["m"].fill_null(False).to_numpy()
EXTRA = {"V1": mask_of(f"{W}/out_descveto_on_v7ens/matching_results.tsv"), "V2": mask_of(f"{W}/out_descveto_gnadd_on_v7ens/matching_results.tsv")}
chk = mask_of("E:/Projects/Amazon ML Challenge/submissions/v7ens/matching_results.tsv")
print("v7ens mask equals model's in_ens:", (chk == M["ens"]).all(), flush=True)
sa = np.isin(nk, ["swap", "added"])
Dgrp = np.isin(cls, ["D", "X"]) & sa & ~(M["ens"] & ~isveto)          # accepted synonym/stem D pairs keep the model p
GNgrp = np.isin(cls, ["G", "N"]) & sa & (num == "nsame") & (F["street"] > 0) & (M["ens"] | M["m"] | (cls == "N"))
print("override groups: D/X", Dgrp.sum(), " G/N same address", GNgrp.sum(), flush=True)
ALL = dict(DIFF)
for k, mk in EXTRA.items():
    add = np.nonzero(mk & ~M["ens"])[0]; rem = np.nonzero(~mk & M["ens"])[0]
    ALL[k] = (add, rem, np.unique(np.concatenate([sidx[add], sidx[rem]])))
PD, PGN = 0.03, 0.96
def pfun(th):
    p = pi_of(th)
    p = np.where(Dgrp, PD, np.where(GNgrp, PGN, p))
    tot = np.bincount(qidx, p, NQ); sc = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
    return p * sc[qidx]
def pred(th):
    p = pfun(th); m = miss_of(p, np.exp(th[8]))
    mask = M["ens"]; pm = np.where(mask, p, 0.0); pr = p - pm
    k = np.bincount(sidx, mask.astype(float), NSC)
    x = np.bincount(sidx, pm, NSC); vx = np.bincount(sidx, pm * (1 - pm), NSC)
    r = np.bincount(sidx, pr, NSC) + m; vr = np.bincount(sidx, pr * (1 - pr), NSC) + m
    p0 = np.exp(np.bincount(sidx, np.log(np.clip(1 - p, 1e-12, 1)), NSC) - m)
    f = MD._f(k, x, vx, r, vr, p0)
    fe = (f.sum() + (NS1 - NSC)) / NS1; E = (p0.sum() + (NS1 - NSC)) / NS1
    out = {"ens": fe, "E": E, "abs": fe - E}
    for kk, (add, rem, us) in ALL.items():
        dk = np.zeros(NSC); dx = np.zeros(NSC); dv = np.zeros(NSC)
        np.add.at(dk, sidx[add], 1); np.add.at(dx, sidx[add], p[add]); np.add.at(dv, sidx[add], p[add] * (1 - p[add]))
        np.add.at(dk, sidx[rem], -1); np.add.at(dx, sidx[rem], -p[rem]); np.add.at(dv, sidx[rem], -p[rem] * (1 - p[rem]))
        f2 = MD._f(k[us] + dk[us], x[us] + dx[us], vx[us] + dv[us], r[us] - dx[us], vr[us] - dv[us], p0[us])
        out[kk] = (f2.sum() - f[us].sum()) / NS1
    out["V2-V1"] = out["V2"] - out["V1"]
    return out
def resid(th, drop=()):
    o = pred(th)
    r = [(o["abs"] - 0.8958) / 0.0015, (o["E"] - 0.056) / 0.006]
    for k, (v, s) in LB.items():
        r.append(0.0 if k in drop else (o[k] - v) / s)
    r += list((th - TH0) / SIG0)
    return np.array(r)
th_full = np.load("C:/ber_scratch/frfix/lbcal/th_full.npy")
def fit(name, drop=()):
    t = time.time()
    res = least_squares(resid, th_full.copy(), diff_step=1e-3, max_nfev=12, ftol=1e-5, xtol=1e-5, kwargs=dict(drop=drop))
    o = pred(res.x); rr = resid(res.x, drop)[:6]
    print(f"[{name} pD={PD} pGN={PGN}] {time.time()-t:.0f}s LB chi2 {np.sum(rr**2):.2f} th {np.round(res.x, 2).tolist()}")
    print("    ", {k: round(v, 5) for k, v in o.items()}, flush=True)
    return res.x
print("no refit, lbcal full-fit params + overrides:", {k: round(v, 5) for k, v in pred(th_full).items()}, flush=True)
import sys as _s
for PD, PGN in [(0.03, 0.96), (0.10, 0.90), (0.30, 0.96), (0.03, 0.80)]:
    fit("all")
    if (PD, PGN) == (0.03, 0.96):
        fit("loo_m", drop=("m",))
