"""Compute and cost: wall-clock and throughput of each step, parsed from the logs in work/, and disk use of work/.

Methodology section 3 ("Cost of the search") and appendix B.6 (wall-clock, throughput, disk).
Reads the run logs WORK/*.log and walks BER_WORK (file sizes, hard links counted once); writes
$BER_SCRATCH/measure/compute_cost.json (default C:/ber_scratch).  python compute_cost.py
"""
import json
import os
import re
import sys
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
from common import WORK

SCR = os.path.join(os.environ.get("BER_SCRATCH", "C:/ber_scratch"), "measure")
os.makedirs(SCR, exist_ok=True)
TS = re.compile(r"^\[\s*([\d.]+)s\]", re.M)


def text(name):
    return open(os.path.join(WORK, name), encoding="utf8", errors="replace").read()


def wall(txt):
    """Sum over runs of the last timestamp of each run (a run restarts when the timestamp drops)."""
    tot, last = 0.0, 0.0
    for x in (float(v) for v in TS.findall(txt)):
        if x < last:
            tot += last
        last = x
    return tot + last


res = {}
# ---- blocking + pair features (laptop CPU): per-chunk "search Xs, features Ys, N pairs"
CH = re.compile(r"search (\d+)s, features (\d+)s, (\d+) pairs")
for label, logs in (("train_build_production", ["full_build.log"]), ("test_build_production", ["test_build.log"]),
                    ("test_build_wide_US_India", ["blocking_b2.log", "blocking_b2_resume.log"])):
    t = "\n".join(text(f) for f in logs)
    ch = [tuple(map(int, m)) for m in CH.findall(t)]
    s, f, n = (sum(c[i] for c in ch) for i in range(3))
    w = sum(wall(text(x)) for x in logs)
    res[label] = {"logs": logs, "chunks": len(ch), "pairs": n, "search_s": s, "features_s": f, "wall_s": round(w),
                  "search_pairs_per_s": round(n / s), "features_pairs_per_s": round(n / f), "pairs_per_wall_s": round(n / w)}

# ---- stage-1 scoring of every searched test pair + stage 2 (LightGBM stage 1 on CPU, one model)
t = text("test_predict.log")
cnt = {c: int(n) for c, n in re.findall(r"(France|India|US): (\d+) candidate pairs", t)}
ts = [float(x) for x in re.findall(r"^\[\s*([\d.]+)s\] (?:France|India|US): \d+ candidate pairs", t, re.M)]
res["test_stage1_scoring_production"] = {"log": "test_predict.log", "pairs": sum(cnt.values()), "stage1_s": ts[-1],
                                         "pairs_per_s": round(sum(cnt.values()) / ts[-1]), "wall_s": wall(t)}
t = text("b2_predict.log")
n = sum(int(x) for x in re.findall(r"parquet: (\d+) pairs", t))
res["test_stage1_scoring_wide"] = {"log": "b2_predict.log", "pairs": n, "wall_s": wall(t), "pairs_per_s": round(n / wall(t)),
                                   "note": "stage 1 + ranks 3-5 + stage 2 of full_cons on the wide pairs"}
for lg, what in (("full_train.log", "stage 1 LightGBM (3 folds + full model) and stage 2 LightGBM, FULL train"),
                 ("v6b_train.log", "stage 2 XGBoost with sibling + consensus features (full_cons), reusing stage 1"),
                 ("topk5_train.log", "stage-1 ranks 3-5 of close-call records, train"),
                 ("topk5_test.log", "stage-1 ranks 3-5 of close-call records, test"),
                 ("embed_train2.log", "fine-tune multilingual-e5-small for the embedding search (276,117 pairs, 1 epoch)"),
                 ("b2_stage3_small.log", "stage 3 XGBoost, e5-small family"), ("b2_stage3_bge.log", "stage 3 XGBoost, bge family"),
                 ("blend_v7p.log", "blend of the two families + rule search"), ("finalize_v7p.log", "decision + writing the TSV")):
    res[lg] = {"what": what, "wall_s": round(wall(text(lg)), 1)}

# ---- transformers
tr = {}
for k in (0, 1, 2):
    t = text(f"ce_folds_small_train_fold_{k}.log")
    steps = re.findall(r"^\[\s*([\d.]+)s\] step (\d+)/(\d+)", t, re.M)
    rows = int(re.search(r"training rows (\d+)", t).group(1))
    tot = int(steps[0][2])
    # rate over steps 0..20000 of the first run (fold 0 was stopped once and resumed from step 36999)
    t0 = float(steps[0][0])
    t20 = next(float(s) for s, st, _ in steps if int(st) == 20000)
    rate = 20000 / (t20 - t0)
    tr[f"e5small_train_fold{k}"] = {"rows": rows, "steps": tot, "batch": 32, "steps_per_s_first_20000": round(rate, 2),
                                    "pairs_per_s": round(rate * 32), "est_train_s_at_this_rate": round(tot / rate)}
    t = text(f"ce_folds_small_score_fold_{k}.log")
    m = re.findall(r"^\[\s*([\d.]+)s\] (train|test): (\d+) close-call rows", t, re.M)
    if len(m) == 2 and "exists, kept" not in t.split("close-call rows")[0] and wall(t) == float(m[1][0]):
        a, b = float(m[0][0]), float(m[1][0])
        tr[f"e5small_score_fold{k}"] = {"train_oof_pairs": int(re.findall(r"train: scored \d+/(\d+)", t)[-1]), "test_pairs": int(m[1][2]),
                                        "test_s": round(b - a, 1), "test_pairs_per_s": round(int(m[1][2]) / (b - a))}
# rescoring of the 585,295 new pairs of the wide search: e5-small (laptop) and bge (RTX A6000)
t = text("b2_score_small.log")
tr["e5small_wide_rescore"] = {"pairs_per_fold": 585295, "s_per_fold": [float(x) for x in
                              re.findall(r"^\[\s*([\d.]+)s\]   test: scored 585295/585295", t, re.M)]}
t = text("b2_bge.log")
tr["bge_wide_rescore_A6000"] = {"pairs_per_fold": 585295, "s_per_fold": [float(x) for x in
                                re.findall(r"\[\s*([\d.]+)s\] test: 3788098 close-call rows", t)]}
for k in ("e5small_wide_rescore", "bge_wide_rescore_A6000"):
    s = tr[k]["s_per_fold"]
    tr[k]["pairs_per_s"] = round(585295 * len(s) / sum(s)) if s else None
t = text("f2_watch.log")
m = re.search(r"remote: (\d\d):(\d\d):(\d\d) OK score fold 1\s+\d\d:\d\d:\d\d START train fold 2", t)
m2 = re.search(r"remote: (\d\d):(\d\d):(\d\d) OK score fold 2", t)
sec = lambda g: int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])
tr["bge_fold2_train_and_score_A6000"] = {"start": ":".join(m.groups()), "end": ":".join(m2.groups()),
                                         "wall_s": sec(m2.groups()) - sec(m.groups())}
t = text("ce_folds_small.log")
c = re.findall(r"^(\d\d):(\d\d):(\d\d) ", t, re.M)
tr["e5small_3folds_wall_incl_failed_tries"] = {"start": ":".join(c[0]), "end": ":".join(c[-1]), "wall_s": sec(c[-1]) - sec(c[0])}
res["transformers"] = tr

# ---- disk: current size of work/ (hard links counted once)
seen, tot, top = set(), 0, {}
for root, dirs, files in os.walk(WORK):
    rel = os.path.relpath(root, WORK).split(os.sep)
    key = rel[0] if rel[0] != "." else "(top-level files)"
    if key == "pairs" and len(rel) > 1:
        key = "pairs/" + rel[1]
    for f in files:
        st = os.stat(os.path.join(root, f))
        ino = (st.st_dev, st.st_ino)
        if ino in seen:
            continue
        seen.add(ino)
        tot += st.st_size
        top[key] = top.get(key, 0) + st.st_size
res["disk_work_GB"] = {"total": round(tot / 1e9, 1)} | {k: round(v / 1e9, 1) for k, v in sorted(top.items(), key=lambda x: -x[1]) if v > 0.5e9}
for tag in ("full", "test", "test_b2"):
    d = os.path.join(WORK, "pairs", tag)
    fs = [os.path.join(d, f) for f in os.listdir(d) if re.fullmatch(r"\w+_\d{3}\.parquet", f)]
    b = sum(os.path.getsize(f) for f in fs)
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
    res["disk_work_GB"][f"pairs_{tag}"] = {"pairs": n, "GB": round(b / 1e9, 1), "bytes_per_pair": round(b / n, 1)}

json.dump(res, open(os.path.join(SCR, "compute_cost.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
