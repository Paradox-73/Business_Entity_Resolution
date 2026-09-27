"""Cross-fitted candidates on the eval half: decision grid, isotonic recalibration, LightGBM stacker."""
import os, sys, time, json, pickle
sys.path.insert(0, r"C:/ber_scratch/final2/usi-stack")
from feats import *
from common import macro_f05_df, log
from pipeline import decide_expf
import numpy as np
import lightgbm as lgb

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
TAG = os.environ.get("TAG", "")
DROP = [f for f in os.environ.get("DROP", "").split(",") if f]
FEATS = [f for f in FEATS if f not in DROP]
d = pl.read_parquet(os.path.join(OUT, "ho.parquet"), columns=["q", "s", "y", "half", "q_noaddr"] + [f for f in FEATS if f != "q_noaddr"])
ev = pl.read_parquet(os.path.join(OUT, "ev.parquet"))
tr = pl.read_parquet(os.path.join(OUT, "truth_ev.parquet"))
SUB = {"all": ev.select("s")}
for h in (0, 1):
    SUB[f"h{h}"] = ev.filter(pl.col("half") == h).select("s")
    for c, cn in ((0, "US"), (1, "IN")):
        SUB[f"h{h}{cn}"] = ev.filter((pl.col("half") == h) & (pl.col("ctry") == c)).select("s")
for c, cn in ((0, "US"), (1, "IN")):
    SUB[cn] = ev.filter(pl.col("ctry") == c).select("s")


def score(pc, floor=0.5, alpha=1.0, name="", pred=None):
    if pred is None:
        pred = decide_expf(d.select("q", "s", p2=pc), "p2", floor, alpha)
    out = {k: macro_f05_df(pred, tr, v) for k, v in SUB.items()}
    return out


def show(name, out, base=None):
    if base is None:
        log(f"{name:34s} " + " ".join(f"{k} {v:.5f}" for k, v in out.items()))
    else:
        log(f"{name:34s} " + " ".join(f"{k} {1e5 * (v - base[k]):+.1f}" for k, v in out.items()) + "  (x1e-5)")


t0 = time.time()
base = score(pl.col("p"))
show("baseline", base)
log(f"decide+score {time.time() - t0:.1f}s")
res = {"base": base}

if MODE in ("all", "grid"):
    # (1) decision grid (global) and per-country rules; choose on one half, report on the other
    grid = {}
    for fl in (0.40, 0.45, 0.5, 0.55, 0.6, 0.65):
        for al in (1.0, 1.25, 1.5):
            o = score(pl.col("p"), fl, al)
            grid[(fl, al)] = o
            show(f"grid floor {fl} alpha {al}", o, base)
    res["grid"] = {f"{k[0]}_{k[1]}": v for k, v in grid.items()}
    # cross-fitted per-country choice: best on half A per country -> applied to half B
    for c in ("US", "IN"):
        for h, oh in ((0, 1), (1, 0)):
            kbest = max(grid, key=lambda k: grid[k][f"h{h}{c}"])
            log(f"per-country {c}: chosen on h{h} {kbest}, gain on h{oh}{c} {1e5 * (grid[kbest][f'h{oh}{c}'] - base[f'h{oh}{c}']):+.1f}e-5")

if MODE in ("all", "iso"):
    from sklearn.isotonic import IsotonicRegression
    pcal = np.zeros(d.height, dtype=np.float32)
    P = d["p"].to_numpy(); Y = d["y"].to_numpy(); H = d["half"].to_numpy(); C = d["ctry"].to_numpy()
    acc = np.zeros(d.height, dtype=np.float32); cnt = np.zeros(d.height, dtype=np.float32)
    for h in (0, 1):
        for c in (0, 1):
            m = (H == h) & (C == c) & ~np.isnan(P)
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[m], Y[m])
            tgt = (H != h) & (C == c) & ~np.isnan(P)
            acc[tgt] += ir.predict(P[tgt]).astype(np.float32); cnt[tgt] += 1
    pcal = np.where(cnt > 0, acc / np.maximum(cnt, 1), P)
    d = d.with_columns(piso=pl.Series(pcal))
    for fl in (0.4, 0.5, 0.6):
        show(f"isotonic per ctry floor {fl}", score(pl.col("piso"), fl, 1.0), base)

if MODE in ("all", "gbdt"):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=6, seed=1)
    X = d.select(FEATS).to_numpy().astype(np.float32)
    Y = d["y"].to_numpy(); H = d["half"].to_numpy()
    preds = {}
    models = {}
    FITON = os.environ.get("FITON", "")
    for h in (0, 1):
        m = (H == h) if not FITON else ((H == -1) & (d["s"].hash(seed=11).to_numpy() % 2 == h))
        ds = lgb.Dataset(X[m], Y[m], feature_name=FEATS, free_raw_data=True)
        t1 = time.time()
        bst = lgb.train(params, ds, num_boost_round=int(os.environ.get("NR", 300)))
        log(f"model h{h}: {m.sum()} rows, {time.time() - t1:.0f}s")
        models[h] = bst
        preds[h] = bst.predict(X).astype(np.float32)
        imp = sorted(zip(FEATS, bst.feature_importance("gain")), key=lambda t: -t[1])[:12]
        log("  top gain: " + ", ".join(f"{k} {v:.3g}" for k, v in imp))
    pst = np.where(H == 0, preds[1], np.where(H == 1, preds[0], 0.5 * (preds[0] + preds[1])))
    d = d.with_columns(pst=pl.Series(pst))
    pickle.dump(models, open(os.path.join(OUT, f"lgb_models{TAG}.pkl"), "wb"))
    for fl, al in ((0.5, 1.0),):
        o = score(pl.col("pst"), fl, al)
        show(f"stacker floor {fl} alpha {al}", o, base)
        res[f"stack_{fl}_{al}"] = o
    pb = decide_expf(d.select("q", "s", p2="p"), "p2", 0.5, 1.0).join(ev.select("s"), on="s")
    pn = decide_expf(d.select("q", "s", p2="pst"), "p2", 0.5, 1.0).join(ev.select("s"), on="s")
    ad = pn.join(pb, on=["s", "q"], how="anti"); rm = pb.join(pn, on=["s", "q"], how="anti")
    ch = pl.concat([ad.select("s"), rm.select("s")]).unique()
    trf = tr.with_columns(t=pl.lit(1))
    log(f"held-out changes: added {ad.height} (true {ad.join(trf, on=['s','q']).height}), removed {rm.height} (true {rm.join(trf, on=['s','q']).height}), rows changed share {ch.height / ev.height:.5f}")
    ad.write_parquet(os.path.join(OUT, f"ho_added{TAG}.parquet")); rm.write_parquet(os.path.join(OUT, f"ho_removed{TAG}.parquet"))
    # blends of stacker with baseline p
    for w in (0.5,):
        o = score((w * pl.col("pst") + (1 - w) * pl.col("p")).cast(pl.Float32), 0.5, 1.0)
        show(f"0.5 stacker + 0.5 p", o, base)
        res["stackmix"] = o
    d.select("q", "s", "half", "p", "pst").write_parquet(os.path.join(OUT, f"ho_pst{TAG}.parquet"))
json.dump(res, open(os.path.join(OUT, f"res_{MODE}{TAG}.json"), "w"), indent=1)
log("done")
