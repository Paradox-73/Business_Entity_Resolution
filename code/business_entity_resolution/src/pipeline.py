"""End-to-end pipeline (v2).

  python pipeline.py build train full        # blocking + features for all of TRAIN -> WORK/pairs/full/
  python pipeline.py build train s10 0.1     # same on a 10% slice (fast ablations)
  python pipeline.py train full              # stage-1 + stage-2 LightGBM, out-of-fold validation, thresholds
  python pipeline.py build test test         # blocking + features for TEST -> WORK/pairs/test/
  python pipeline.py predict full test       # apply models trained on `full` -> output/*.tsv

Validation: 3 folds grouped by true S1 (all S2/S3 rows of one business share a fold). Scores are
reported on the S1 rows OUTSIDE the embedding fine-tuning partition (see embed.emb_partition), and
stage-1/2 models are trained only on rows whose true S1 is in that held-out half (or who have no match),
so the learned-embedding feature never leaks.

Decision: each S2/S3 row goes to its single best S1 (by stage-2 probability) if that probability is at
least the threshold of its segment (address missing? non-Latin name? source?), else to nobody.
Thresholds are tuned on out-of-fold predictions with the competition metric.
"""
import glob
import json
import os
import sys
import zlib
import lightgbm as lgb
import numpy as np
import polars as pl
from common import WORK, OUT, log, read_truth, id_to_int, int_to_id, macro_f05_df
from candidates import FEATURES, country_pairs

# BER_S1_VARIANT selects the pair-feature set for stage 1 AND the pair part of stage 2 (models go to <dir>_<variant>).
# "nomargin" (audit 25 Sep): drop every feature that describes a record's OTHER candidates (gaps to the next-best
# candidate, ranks, candidate counts, chain count). They depend on S1 density and on how many look-alikes exist,
# both of which differ between train and test; v4 showed the model is very sensitive to them.
S1_VARIANT = os.environ.get("BER_S1_VARIANT", "")
# BER_S2_ELIG=latin: stage 2 also trains on records of the embedding fine-tuning half whose name is Latin (the
# embedding leak the eval-half rule guards against only exists for non-Latin names). BER_S2_RATE overrides the
# share of records used per fold. Output dir gets the suffix BER_S2_TAG (e.g. models/full_sib_all).
S2_ELIG = os.environ.get("BER_S2_ELIG", "")
S2_TAG = os.environ.get("BER_S2_TAG", "")
DROP1 = {"nomargin": [f for f in FEATURES if f.endswith("_margin") or f.endswith("_rank")]
                     + ["n_cand", "n_name_hi", "n_addr_hi", "s_name_count"]}
FEATURES1 = [f for f in FEATURES if f not in DROP1.get(S1_VARIANT, [])]

N_FOLDS = 3
PARAMS1 = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
               feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
               num_threads=11, verbose=-1, seed=7)
ROUNDS1 = 600
PARAMS2 = dict(PARAMS1, num_leaves=127, learning_rate=0.05)
ROUNDS2 = 500
# model backend: "xgb" = XGBoost on the GPU (default; ~8x faster to score than CPU LightGBM), "lgb" = CPU LightGBM.
# XGBoost 2.0.3 is pinned: it is the last build that runs on CUDA 11 drivers (this laptop: driver 511, CUDA 11.6).
BACKEND = os.environ.get("BER_BACKEND", "xgb")          # stage 1
BACKEND2 = os.environ.get("BER_BACKEND2", BACKEND)      # stage 2 (can differ, e.g. reuse a cached LightGBM stage 1)
XGB_PARAMS = {1: dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.08, max_depth=10,
                      min_child_weight=20, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=7),
              2: dict(tree_method="hist", device="cuda", objective="binary:logistic", eta=0.05, max_depth=8,
                      min_child_weight=20, subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=7)}
XGB_ROUNDS = {1: 1000, 2: 500}   # v4 capacity test: deeper (depth 10) + more rounds than LightGBM stage 1
XGB_PARAMS[1]["eta"] = 0.06
EXT = ".json" if BACKEND == "xgb" else ".txt"
EXT2 = ".json" if BACKEND2 == "xgb" else ".txt"


def fit(stage, Xm, y):
    """Train one stage-1 or stage-2 model on feature matrix Xm and labels y: XGBoost on the GPU or LightGBM
    (BER_BACKEND for stage 1, BER_BACKEND2 for stage 2). Returns the booster."""
    if (BACKEND if stage == 1 else BACKEND2) == "xgb":
        import xgboost as xgb
        return xgb.train(XGB_PARAMS[stage], xgb.QuantileDMatrix(Xm, y), XGB_ROUNDS[stage])
    return lgb.train(PARAMS1 if stage == 1 else PARAMS2, lgb.Dataset(Xm, y), ROUNDS1 if stage == 1 else ROUNDS2)


def score(m, Xm):
    """Probabilities of model m (LightGBM or XGBoost) for the rows of Xm; XGBoost scores 2M rows at a time
    to stay within 4 GB of GPU memory."""
    if isinstance(m, lgb.Booster):
        return m.predict(Xm)
    import xgboost as xgb
    out = np.empty(len(Xm), np.float32)
    for a in range(0, len(Xm), 2_000_000):          # batches keep GPU memory under 4 GB
        out[a:a + 2_000_000] = m.predict(xgb.DMatrix(Xm[a:a + 2_000_000]))
    return out


def load_model(path_noext):
    """Load a saved booster: <path_noext>.json (XGBoost, set to the GPU) if it exists, else <path_noext>.txt
    (LightGBM)."""
    if os.path.exists(path_noext + ".json"):
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(path_noext + ".json")
        m.set_param({"device": "cuda"})
        return m
    return lgb.Booster(model_file=path_noext + ".txt")


def importance(models, cols):
    """Split gain per feature summed over `models`, as (feature, gain) pairs from largest to smallest."""
    if isinstance(models[0], lgb.Booster):
        return sorted(zip(cols, sum(m.feature_importance("gain") for m in models)), key=lambda x: -x[1])
    tot = {}
    for m in models:
        for k, v in m.get_score(importance_type="total_gain").items():
            tot[cols[int(k[1:])]] = tot.get(cols[int(k[1:])], 0) + v
    return sorted(tot.items(), key=lambda x: -x[1])
TOPK2 = 2
# stage-1 training sample (rows, not whole records: group features are already computed at build time):
# every matching pair + a share of "hard" non-matches (top-2 by name or address, or look-alike names)
# + a share of the rest. Keeps ~8M rows for 16 GB RAM. (exp 6: record-level 6% sampling kept only 182k positives)
NEG_RATE = {"full": (0.15, 0.015), "s10": (1.0, 0.10)}
STAGE2_RATE = {"full": 0.5, "s10": 1.0, "tlike": 0.5, "tl2": 0.5}
SEG_COLS = ["addr_missing", "name_nonlatin", "src"]
SEG_PASSES = 0   # per-segment threshold search: 0 = off (gave +0.0000 on s10, exp 5) and is slow at full scale
S2FEATS = ["p1", "p1_rank", "p1_margin", "s_in_cnt", "s_in_sum", "s_in_max_other", "s_in_same_src",
           "q_rank_in_s", "s_base_cnt"]


def load(split, k, cols=None):
    """Cleaned source file WORK/<split>_s<k>.parquet (k = 1: S1 rows; 2, 3: records), optionally only `cols`."""
    return pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"), columns=cols)


def crc(x):
    """zlib.crc32 of an id string. For an S1 row, crc % 1000 < 500 marks the held-out half and crc % 3 its fold."""
    return zlib.crc32(x.encode())


def pairs_dir(tag):
    """Folder of the pair-feature chunk files of one build: WORK/pairs/<tag>."""
    return os.path.join(WORK, "pairs", tag)


# ---------------------------------------------------------------- build
def build(split, tag, frac=None):
    """Blocking and the 55 pair features for one split. Writes WORK/pairs/<tag>/s1.parquet, q.parquet and one
    <country>_<chunk>.parquet per chunk of records. frac (e.g. 0.1) keeps that share of the S1 rows with their true records,
    plus the same share of unmatched records. Countries marked done and chunk files already on disk are skipped."""
    out = pairs_dir(tag)
    os.makedirs(out, exist_ok=True)
    s1 = load(split, 1)
    q = pl.concat([load(split, k) for k in (2, 3)])
    if frac:
        truth = read_truth()
        s1 = s1.filter(pl.col("entity_id").map_elements(lambda x: crc(x) % 1000 < frac * 1000, return_dtype=pl.Boolean))
        keep = set(truth.filter(pl.col("s1_id").is_in(s1["entity_id"].to_list()))["q_id"])
        matched = set(truth["q_id"])
        q = q.filter(pl.col("entity_id").is_in(list(keep)) |
                     (~pl.col("entity_id").is_in(list(matched)) &
                      pl.col("entity_id").map_elements(lambda x: crc(x) % 1000 < frac * 1000, return_dtype=pl.Boolean)))
    s1.select(s=id_to_int("entity_id"), s1_id="entity_id", country="country").write_parquet(os.path.join(out, "s1.parquet"))
    q.select(q=id_to_int("entity_id"), country="country", src="src", addr_missing="addr_missing",
             name_nonlatin="name_nonlatin").write_parquet(os.path.join(out, "q.parquet"))
    log(f"build {split}/{tag}: S1 {s1.height}, S2/S3 {q.height}")
    for c in s1["country"].unique().sort().to_list():
        done = os.path.join(out, f"{c}.done")
        if os.path.exists(done):
            continue
        log(f"country {c}")

        def on_chunk(p, i):
            """Save chunk i of the current country: integer ids q, s and the float32 features."""
            p = (p.with_columns(q=id_to_int("entity_id"), s=id_to_int("entity_id_s"))
                  .select(["q", "s"] + [pl.col(f).cast(pl.Float32) for f in FEATURES]))
            p.write_parquet(os.path.join(out, f"{c}_{i:03d}.parquet"))

        country_pairs(s1.filter(pl.col("country") == c), q.filter(pl.col("country") == c), split, on_chunk,
                      skip=lambda i: os.path.exists(os.path.join(out, f"{c}_{i:03d}.parquet")))
        open(done, "w").close()


def chunk_files(tag):
    """Sorted paths of the pair-feature chunk files of WORK/pairs/<tag>."""
    return sorted(f for f in glob.glob(os.path.join(pairs_dir(tag), "*_*.parquet")))


# ---------------------------------------------------------------- labels / folds
def train_meta(tag):
    """S1 table with fold + eval flag; q->true s map; truth pairs (ints)."""
    s1 = pl.read_parquet(os.path.join(pairs_dir(tag), "s1.parquet"))
    s1 = s1.with_columns(h=pl.col("s1_id").map_elements(crc, return_dtype=pl.Int64)).with_columns(
        fold=(pl.col("h") % N_FOLDS).cast(pl.Int8), is_eval=(pl.col("h") % 1000 < 500)).drop("h")
    t = read_truth().select(s=id_to_int("s1_id"), q=id_to_int("q_id")).join(s1.select("s"), on="s")
    qmap = t.join(s1.select("s", "fold", "is_eval"), on="s").select("q", true_s="s", qfold="fold", q_eval="is_eval")
    return s1, t, qmap


def attach(p, qmap, rate=None):
    """Add to pairs p: label (s is the record's true S1 row), fold (fold of the true S1 row; q % 3 for unmatched
    records) and eligible (the true S1 row is in the held-out half, or the record has none). With rate, keep only
    eligible records, sampled at that rate."""
    p = (p.join(qmap, on="q", how="left")
          .with_columns(label=(pl.col("true_s") == pl.col("s")).fill_null(False),
                        fold=pl.coalesce("qfold", (pl.col("q") % N_FOLDS).cast(pl.Int8)),
                        eligible=pl.col("q_eval").fill_null(True)))
    if rate is not None:
        p = p.filter(pl.col("eligible") & ((pl.col("q").hash(seed=1) % 1000) < rate * 1000))
    return p


def sample1(p, qmap, hard_rate, rand_rate):
    """Stage-1 training rows of one chunk: every eligible true pair, hard negatives (rank <= 2 by name cosine or
    address token-set, or name token-set >= 80) at hard_rate, and other rows at rand_rate."""
    p = attach(p, qmap)
    h = pl.struct("q", "s").hash(seed=3) % 10000
    hard = (pl.col("cos_name_rank") <= 2) | (pl.col("a_tset_rank") <= 2) | (pl.col("n_tset") >= 80)
    return p.filter(pl.col("eligible") & (pl.col("label") | (hard & (h < hard_rate * 10000)) |
                                          (h < rand_rate * 10000)))


def X(df, cols):
    """Columns `cols` of df as a float32 numpy matrix (model input)."""
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def topk(p):
    """Best TOPK2 candidates per q by stage-1 prob; lower-ranked ones only if p1 >= 0.01 (saves memory)."""
    t = p.sort("p1", descending=True).group_by("q", maintain_order=True).head(TOPK2)
    return t.filter((pl.col("p1").rank("ordinal", descending=True).over("q") == 1) | (pl.col("p1") >= 0.01))


# ---------------------------------------------------------------- stage-2 features
def stage2_features(b, qinfo):
    """b: top-TOPK2 rows per q with p1. Adds competition features from the S1 side."""
    b = b.with_columns(p1_rank=pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Float32))
    b = b.with_columns(p1_margin=pl.col("p1") - pl.when(pl.len().over("q") > 1)
                       .then(pl.col("p1").sum().over("q") - pl.col("p1")).otherwise(0.0))
    top = b.filter(pl.col("p1_rank") == 1)
    hi = (pl.col("p1") >= 0.5).cast(pl.Float32)
    g = top.group_by("s").agg(cnt=hi.sum(), sm=pl.col("p1").sum(), m1=pl.col("p1").max(),
                              m2=pl.col("p1").top_k(2).min(), n=pl.len())
    gs = top.group_by("s", "src").agg(cs=hi.sum())
    b = b.join(g, on="s", how="left").join(gs, on=["s", "src"], how="left").with_columns(
        [pl.col(c).fill_null(0) for c in ("cnt", "sm", "m1", "m2", "n", "cs")])
    self_top = (pl.col("p1_rank") == 1)
    self_hi = (self_top & (pl.col("p1") >= 0.5)).cast(pl.Float32)
    b = b.with_columns(
        s_in_cnt=pl.col("cnt") - self_hi,
        s_in_sum=pl.col("sm") - pl.when(self_top).then(pl.col("p1")).otherwise(0.0),
        s_in_max_other=pl.when(self_top & (pl.col("p1") == pl.col("m1")) & (pl.col("n") > 1)).then(pl.col("m2"))
                         .when(self_top & (pl.col("n") == 1)).then(0.0).otherwise(pl.col("m1")),
        s_in_same_src=pl.col("cs") - self_hi,
        q_rank_in_s=pl.col("p1").rank("ordinal", descending=True).over("s").cast(pl.Float32),
        s_base_cnt=pl.len().over("s").cast(pl.Float32),
    ).drop("cnt", "sm", "m1", "m2", "n", "cs")
    return b


SIBFEATS = ["sib_p1", "sib_name_tset", "sib_addr_tset", "sib_num_eq", "sib_same_src", "sib_missing"]


def load_qattr(split):
    """q (int) -> name_core, addr, first house number, for S2+S3 rows (sibling features)."""
    return pl.concat([pl.read_parquet(os.path.join(WORK, f"{split}_s{k}.parquet"),
                                      columns=["entity_id", "name_core", "addr", "addr_nums"]) for k in (2, 3)]).select(
        q=id_to_int("entity_id"), nm="name_core", ad="addr", n1=pl.col("addr_nums").list.first())


def sibling_features(b, qattr):
    """Compare each row q with its 'sibling': the best OTHER S2/S3 row whose top candidate is the same S1.
    A true match usually agrees with its siblings (they describe the same business); a decoy does not."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    top = b.filter(pl.col("p1_rank") == 1).select("s", "q", "p1")
    t2 = (top.sort("p1", descending=True).group_by("s", maintain_order=True).head(2)
             .with_columns(r=pl.int_range(pl.len()).over("s")))
    first = t2.filter(pl.col("r") == 0).select("s", qa="q", pa="p1")
    second = t2.filter(pl.col("r") == 1).select("s", qb="q", pb="p1")
    b = b.join(first, on="s", how="left").join(second, on="s", how="left").with_columns(
        sib_q=pl.when(pl.col("qa") == pl.col("q")).then(pl.col("qb")).otherwise(pl.col("qa")),
        sib_p1=pl.when(pl.col("qa") == pl.col("q")).then(pl.col("pb")).otherwise(pl.col("pa"))).drop("qa", "qb", "pa", "pb")
    b = (b.join(qattr, on="q", how="left")
          .join(qattr.rename({"q": "sib_q", "nm": "nm_s", "ad": "ad_s", "n1": "n1_s"}), on="sib_q", how="left"))
    fill = lambda c: b[c].fill_null("").to_list()
    ns = cpdist(fill("nm"), fill("nm_s"), scorer=fuzz.token_set_ratio, workers=int(os.environ.get("BER_THREADS", "-1")), dtype=np.float32)
    ad = cpdist(fill("ad"), fill("ad_s"), scorer=fuzz.token_set_ratio, workers=int(os.environ.get("BER_THREADS", "-1")), dtype=np.float32)
    miss = b["sib_q"].is_null()
    b = b.with_columns(
        sib_name_tset=pl.when(miss).then(None).otherwise(pl.Series(ns)),
        sib_addr_tset=pl.when(miss).then(None).otherwise(pl.Series(ad)),
        sib_num_eq=(pl.col("n1") == pl.col("n1_s")).cast(pl.Float32),
        sib_same_src=((pl.col("q") // 10_000_000_000) == (pl.col("sib_q") // 10_000_000_000)).cast(pl.Float32),
        sib_missing=miss.cast(pl.Float32),
    ).drop("nm", "ad", "n1", "nm_s", "ad_s", "n1_s", "sib_q")
    return b


CONSFEATS = ["cons_addr_mean", "cons_addr_min", "cons_name_mean", "cons_name_min", "cons_num_share", "cons_num_same"]


def consensus_features(b, qattr, parts=8):
    """Compare each row (q, s) with ALL other S2/S3 rows whose top candidate is the same S1 with p1 >= 0.5
    (sibling_features uses only the best one): mean/min name and address agreement, and how many of them
    share q's first house number. A true match agrees with the group; a look-alike (shifted number) does not.
    Processed in `parts` slices of S1 to bound memory."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    mem = (b.filter((pl.col("p1_rank") == 1) & (pl.col("p1") >= 0.5)).select("s", qm="q")
            .join(qattr.rename({"q": "qm", "nm": "nm_m", "ad": "ad_m", "n1": "n1_m"}), on="qm", how="left"))
    key = b.select("q", "s").with_columns(h=(pl.col("s").hash(seed=7) % parts))
    out = []
    for i in range(parts):
        x = (key.filter(pl.col("h") == i).drop("h").join(mem, on="s").filter(pl.col("qm") != pl.col("q"))
                .join(qattr, on="q", how="left"))
        fill = lambda c: x[c].fill_null("").to_list()
        ns = cpdist(fill("nm"), fill("nm_m"), scorer=fuzz.token_set_ratio, workers=int(os.environ.get("BER_THREADS", "-1")), dtype=np.float32)
        ad = cpdist(fill("ad"), fill("ad_m"), scorer=fuzz.token_set_ratio, workers=int(os.environ.get("BER_THREADS", "-1")), dtype=np.float32)
        no_ad = (pl.col("ad").fill_null("") == "") | (pl.col("ad_m").fill_null("") == "")
        x = x.with_columns(ns=pl.Series(ns), ad=pl.when(no_ad).then(None).otherwise(pl.Series(ad)),
                           eq=pl.when(pl.col("n1").is_null() | pl.col("n1_m").is_null()).then(None)
                              .otherwise((pl.col("n1") == pl.col("n1_m")).cast(pl.Float32)))
        out.append(x.group_by("q", "s").agg(
            cons_addr_mean=pl.col("ad").mean(), cons_addr_min=pl.col("ad").min(),
            cons_name_mean=pl.col("ns").mean(), cons_name_min=pl.col("ns").min(),
            cons_num_share=pl.col("eq").mean(), cons_num_same=pl.col("eq").sum()))
        del x
    return b.join(pl.concat(out), on=["q", "s"], how="left")


# ---------------------------------------------------------------- decision
def decide(b, prob, thr, qinfo):
    """Top-1 S1 per q by `prob`, kept if prob >= its segment threshold. Returns DataFrame s, q."""
    top = b.sort(prob, descending=True).unique("q", keep="first").join(qinfo, on="q", how="left")
    seg = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in SEG_COLS], separator="|")
    t = seg.replace_strict(thr, default=thr.get("default", 0.5), return_dtype=pl.Float64) if isinstance(thr, dict) \
        else pl.lit(thr)
    return top.filter(pl.col(prob) >= t).select("s", "q")


def decide_expf(b, prob, floor, alpha):
    """Per S1: among S2/S3 rows whose best S1 it is (prob >= floor), pick the top-k set that maximises
    expected F0.5 ~ 1.25*sum(p_1..k) / (k + 0.25*sum(p_all)), versus predicting nothing (prod(1-p)).
    Returns DataFrame s, q."""
    c = (b.sort(prob, descending=True).unique("q", keep="first").filter(pl.col(prob) >= floor)
          .sort(["s", prob], descending=[False, True]).with_columns(pp=pl.col(prob) ** alpha))
    c = c.with_columns(k=pl.int_range(1, pl.len() + 1).over("s"), cum=pl.col("pp").cum_sum().over("s"),
                       tot=pl.col("pp").sum().over("s"),
                       none=(1 - pl.col("pp")).clip(1e-9, 1).log().sum().over("s").exp())
    c = c.with_columns(ef=1.25 * pl.col("cum") / (pl.col("k") + 0.25 * pl.col("tot")))
    best = c.group_by("s").agg(bk=pl.col("k").sort_by("ef", descending=True).first(), bef=pl.col("ef").max())
    return (c.join(best, on="s").filter((pl.col("k") <= pl.col("bk")) & (pl.col("bef") > pl.col("none")))
             .select("s", "q"))


def apply_decision(b, prob, dec, qinfo):
    """Matches (s, q) from probability column `prob` with a saved rule: expected-F0.5 sets (dec type 'expf')
    or thresholds (type 'thr', one value or one per segment)."""
    if dec["type"] == "expf":
        return decide_expf(b, prob, dec["floor"], dec["alpha"])
    return decide(b, prob, dec["t"], qinfo)


def tune_decision(b, prob, truth, s1e, g, gscore):
    """Compare the global threshold g with expected-F0.5 set selection; return the better rule."""
    best, dec = gscore, {"type": "thr", "t": g}
    for floor in (0.3, 0.5):
        for alpha in (1.0, 1.5):
            sc = macro_f05_df(decide_expf(b, prob, floor, alpha), truth, s1e)
            log(f"   expF floor={floor} alpha={alpha}: {sc:.5f}")
            if sc > best + 1e-6:
                best, dec = sc, {"type": "expf", "floor": floor, "alpha": alpha}
    return dec, best


def tune(b, prob, truth, s1e, qinfo):
    """Global threshold, then per-segment coordinate ascent. Returns (thresholds, score, global_score)."""
    grid = [round(x, 3) for x in np.arange(0.3, 0.96, 0.025)]
    evalset = set(s1e["s"].to_list())
    bb = b.filter(pl.col("s").is_in(list(evalset)) | pl.col("q").is_in(truth["q"].to_list()))

    def score(thr):
        """Held-out macro F0.5 of the threshold rule `thr` (one value or a per-segment dict)."""
        return macro_f05_df(decide(bb, prob, thr, qinfo), truth, s1e)

    gs = {t: score(t) for t in grid}
    g = max(gs, key=gs.get)
    segs = (qinfo.select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in SEG_COLS], separator="|"))
                 .to_series().unique().to_list())
    thr = {s: g for s in segs}
    thr["default"] = g
    best = gs[g]
    for _ in range(SEG_PASSES):
        for s in segs:
            for t in grid:
                cand = dict(thr, **{s: t})
                sc = score(cand)
                if sc > best + 1e-6:
                    best, thr = sc, cand
    return thr, best, (g, gs[g])


# ---------------------------------------------------------------- train
def train(tag, variant=""):
    """variant: '' = base stage 2; 'sib' = stage 2 with sibling features. Stage 1 is shared (cached per tag);
    stage-2 outputs of a variant go to models/<tag>_<variant>/ so the base model stays untouched."""
    s1, t, qmap = train_meta(tag)
    qinfo = pl.read_parquet(os.path.join(pairs_dir(tag), "q.parquet")).drop("country")
    files = chunk_files(tag)
    rate = NEG_RATE.get(tag, (0.15, 0.015))
    s1e = s1.filter("is_eval").select("s")
    te = t.join(s1e, on="s")
    log(f"train {tag}: {len(files)} chunk files, eval S1 {s1e.height}, eval true pairs {te.height}")

    md = os.path.join(WORK, "models", tag + ("_xgb" if BACKEND == "xgb" else "") + (f"_{S1_VARIANT}" if S1_VARIANT else ""))
    os.makedirs(md, exist_ok=True)
    cache = os.path.join(md, "stage1_base.parquet")
    if os.path.exists(cache):
        # stage 1 already done for this tag: reuse models + top-K table
        models1 = [load_model(os.path.join(md, f"s1_f{i}")) for i in range(N_FOLDS)]
        B = pl.read_parquet(cache)
        s1stats = json.load(open(os.path.join(md, "stage1.json")))
        found, npairs = s1stats["found"], s1stats["npairs"]
        nq = qinfo.height
        g1, gs1, sc1 = s1stats["g1"], s1stats["gs1"], s1stats["sc1"]
        log(f"reusing stage 1 from {cache}: OOF {gs1:.4f}")
    else:
        B, models1, found, npairs, g1, gs1, sc1 = stage1(files, qmap, rate, s1e, te, qinfo, md)
        nq = qinfo.height
    md2 = (f"{md}_{variant}" if variant else md) + (f"_{S2_TAG}" if S2_TAG else "")
    os.makedirs(md2, exist_ok=True)
    return train_stage2(tag, B, models1, found, npairs, nq, te, s1e, qinfo, md2, g1, gs1, sc1, variant, md)


def stage1(files, qmap, rate, s1e, te, qinfo, md):
    # ---- stage 1: training sample
    """Stage 1: trains 3 fold models (and, for LightGBM, one model on the whole sample) on the sampled pairs,
    scores every pair out-of-fold, keeps each record's top candidates (topk) and tunes a threshold on the held-out half.
    Writes s1_f<k>, s1_full, stage1_base.parquet and stage1.json to md; returns the top-candidate table, the fold models
    and the recall and score figures."""
    parts = [sample1(pl.read_parquet(f), qmap, *rate) for f in files]
    S = pl.concat(parts)
    del parts
    log(f"stage-1 sample: {S.height} rows, positives {int(S['label'].sum())}")
    models1 = []
    for f in range(N_FOLDS):
        tr = S.filter(pl.col("fold") != f)
        models1.append(fit(1, X(tr, FEATURES1), tr["label"].to_numpy()))
        models1[-1].save_model(os.path.join(md, f"s1_f{f}" + EXT))   # save at once (crash safety)
        log(f"stage-1 fold {f} trained on {tr.height}")
    imp = importance(models1, FEATURES1)
    log("stage-1 top features:", [(a, int(b)) for a, b in imp[:20]])
    del tr
    # one model on the whole sample, used for TEST (3x cheaper to apply than averaging fold models)
    if BACKEND != "xgb":   # GPU (4 GB) cannot hold the full 11.5M-row sample; XGBoost test uses the fold-model average
        fit(1, X(S, FEATURES1), S["label"].to_numpy()).save_model(os.path.join(md, "s1_full" + EXT))
    log("stage-1 full-sample model saved")
    del S

    # ---- stage 1: OOF over every pair; keep top-K per q
    base, found, npairs = [], 0, 0
    evs = s1e["s"]
    for fpath in files:
        p = attach(pl.read_parquet(fpath), qmap)
        npairs += p.height
        found += int(p.filter(pl.col("label") & pl.col("s").is_in(evs.implode())).height)
        p1 = np.zeros(p.height, np.float32)
        fold = p["fold"].to_numpy()
        for f in range(N_FOLDS):
            m = fold == f
            if m.any():
                p1[m] = score(models1[f], X(p.filter(pl.Series(m)), FEATURES1))
        p = p.with_columns(p1=pl.Series(p1))
        base.append(topk(p))
    B = pl.concat(base)
    del base
    nq = qinfo.height
    log(f"pairs {npairs} ({npairs / nq:.1f} per S2/S3 row); recall ceiling on eval S1: {found / te.height:.4f}")
    th1, sc1, (g1, gs1) = tune(B, "p1", te, s1e, qinfo)
    log(f"STAGE-1 OOF macro F0.5: global t={g1} -> {gs1:.4f}; segment thresholds -> {sc1:.4f}")
    for i, m in enumerate(models1):
        m.save_model(os.path.join(md, f"s1_f{i}" + EXT))
    B.write_parquet(os.path.join(md, "stage1_base.parquet"))
    json.dump({"found": found, "npairs": npairs, "g1": g1, "gs1": gs1, "sc1": sc1, "th1": th1},
              open(os.path.join(md, "stage1.json"), "w"), indent=1)
    return B, models1, found, npairs, g1, gs1, sc1


def train_stage2(tag, B, models1, found, npairs, nq, te, s1e, qinfo, md, g1, gs1, sc1, variant="", md1=None):
    # ---- stage 2
    """Stage 2 on the stage-1 top candidates B: adds the S1-side competition features (and the sibling /
    consensus features for variant 'sib' / 'cons'), trains 3 fold models, scores out-of-fold and tunes the decision rule.
    Writes s2_f<k>, result.json (features, rule, scores) and oof.parquet (q, s, p1, p2, label, fold) to md."""
    B = stage2_features(B, qinfo)
    cols2 = FEATURES1 + S2FEATS
    if variant in ("sib", "cons"):
        qattr = load_qattr("train")
        B = sibling_features(B, qattr)
        cols2 = cols2 + SIBFEATS
        log("sibling features added")
        if variant == "cons":
            B = consensus_features(B, qattr)
            cols2 = cols2 + CONSFEATS
            log("consensus features added")
        del qattr
    p2 = np.zeros(B.height, np.float32)
    models2 = []
    elig = pl.col("eligible")
    if S2_ELIG == "latin":
        B = B.join(qinfo.select("q", _nl="name_nonlatin"), on="q", how="left")
        elig = pl.col("eligible") | ~pl.col("_nl").fill_null(True)
    rate = float(os.environ.get("BER_S2_RATE", STAGE2_RATE.get(tag, 1.0)))
    log(f"stage-2 training rows: eligibility {S2_ELIG or 'eval half'}, rate {rate}")
    for f in range(N_FOLDS):
        tr = B.filter((pl.col("fold") != f) & elig &
                      ((pl.col("q").hash(seed=5) % 1000) < rate * 1000))
        m = fit(2, X(tr, cols2), tr["label"].to_numpy())
        models2.append(m)
        msk = (B["fold"] == f).to_numpy()
        p2[msk] = score(m, X(B.filter(pl.Series(msk)), cols2))
        log(f"stage-2 fold {f} trained on {tr.height}")
    B = B.with_columns(p2=pl.Series(p2))
    imp2 = importance(models2, cols2)
    log("stage-2 top features:", [(a, int(b)) for a, b in imp2[:15]])
    th2, sc2, (g2, gs2) = tune(B, "p2", te, s1e, qinfo)
    log(f"STAGE-2 OOF macro F0.5: global t={g2} -> {gs2:.4f}; segment thresholds -> {sc2:.4f}")
    dec, sc_dec = tune_decision(B, "p2", te, s1e, g2, gs2)
    log(f"DECISION: {dec} -> {sc_dec:.5f}")

    for i, m in enumerate(models2):
        m.save_model(os.path.join(md, f"s2_f{i}" + EXT2))
    tr = B.filter(pl.col("eligible"))
    if BACKEND2 != "xgb":
        fit(2, X(tr, cols2), tr["label"].to_numpy()).save_model(os.path.join(md, "s2_full" + EXT2))
    del tr
    res = {"tag": tag, "recall_ceiling": found / te.height, "pairs_per_q": npairs / nq,
           "stage1_global": [g1, gs1], "stage1_seg": sc1, "stage2_global": [g2, gs2], "stage2_seg": sc2,
           "thresholds": th2, "use_stage2": sc2 >= sc1, "decision": dec, "decision_score": sc_dec,
           "cols2": cols2, "cols1": FEATURES1, "stage1_dir": md1 or md}
    json.dump(res, open(os.path.join(md, "result.json"), "w"), indent=1)
    B.select("q", "s", "p1", "p2", "label", "fold").write_parquet(os.path.join(md, "oof.parquet"))
    log("saved", md, res)


# ---------------------------------------------------------------- predict
def predict(model_tag, test_tag):
    """Score every test pair of WORK/pairs/<test_tag> with the saved models of WORK/models/<model_tag>.
    Writes WORK/test_scores_<model_tag>[_<test_tag>].parquet (q, s, p1, p2) and, in BER_OUT, matching_results.tsv (saved
    rule) and candidate_pairs.tsv (every searched pair). Stage-1 test scores are cached in WORK/test_base_*.parquet."""
    md = os.path.join(WORK, "models", model_tag)
    res = json.load(open(os.path.join(md, "result.json")))
    def models(stage):
        """Saved models of one stage: the full-sample model if it exists, else the 3 fold models (averaged later)."""
        d = res.get("stage1_dir", md) if stage == 1 else md
        full = os.path.join(d, f"s{stage}_full")
        if os.path.exists(full + ".json") or os.path.exists(full + ".txt"):
            return [load_model(full)]
        return [load_model(os.path.join(d, f"s{stage}_f{i}")) for i in range(N_FOLDS)]
    m1, m2 = models(1), models(2)
    cols1 = res.get("cols1", FEATURES)
    log(f"test models: stage 1 x{len(m1)}, stage 2 x{len(m2)}")
    d = pairs_dir(test_tag)
    s1 = pl.read_parquet(os.path.join(d, "s1.parquet"))
    qinfo = pl.read_parquet(os.path.join(d, "q.parquet")).drop("country")
    os.makedirs(OUT, exist_ok=True)
    # stage-1 test scores depend only on the stage-1 model: cache them so a new stage 2 re-predicts in minutes
    s1tag = os.path.basename(res.get("stage1_dir", md))
    base_cache = os.path.join(WORK, f"test_base_{test_tag}_{s1tag}.parquet")
    cand_cache = os.path.join(WORK, f"test_cands_{test_tag}.parquet")
    if os.path.exists(base_cache) and os.path.exists(cand_cache):
        B0, C = pl.read_parquet(base_cache), pl.read_parquet(cand_cache)
        log(f"reusing stage-1 test scores {base_cache}")
    else:
        base, cands = [], []
        for c in s1["country"].unique().sort().to_list():
            pairs = []
            for fpath in sorted(glob.glob(os.path.join(d, f"{c}_*.parquet"))):
                p = pl.read_parquet(fpath)
                p = p.with_columns(p1=pl.Series(np.mean([score(m, X(p, cols1)) for m in m1], axis=0), dtype=pl.Float32))
                base.append(topk(p))
                pairs.append(p.select("s", "q"))
            P = pl.concat(pairs)
            cands.append(P.group_by("s").agg(pl.col("q").unique().sort()))
            log(f"{c}: {P.height} candidate pairs")
            del P, pairs
        B0, C = pl.concat(base), pl.concat(cands)
        del base, cands
        B0.write_parquet(base_cache)
        C.write_parquet(cand_cache)
    B = stage2_features(B0, qinfo)
    del B0
    cols2 = res.get("cols2", FEATURES + S2FEATS)
    if any(c in cols2 for c in SIBFEATS):
        # sibling text must be cleaned the same way as the pairs (e.g. 'testfr' = the countries without training
        # labels, cleaned with the French rules)
        split = test_tag if os.path.exists(os.path.join(WORK, f"{test_tag}_s2.parquet")) else "test"
        qattr = load_qattr(split)
        B = sibling_features(B, qattr)
        if any(c in cols2 for c in CONSFEATS):
            B = consensus_features(B, qattr)
        del qattr
    B = B.with_columns(p2=pl.Series(np.mean([score(m, X(B, cols2)) for m in m2], axis=0), dtype=pl.Float32))
    prob = "p2" if res["use_stage2"] else "p1"
    dec = res["decision"] if res["use_stage2"] else {"type": "thr", "t": res["stage1_global"][0]}
    M = apply_decision(B, prob, dec, qinfo)
    log(f"decision {prob} {dec}: {M.height} matches")
    sfx = "" if test_tag == "test" else f"_{test_tag}"      # e.g. the 'testfr' rebuild must not overwrite full-test scores
    B.select("q", "s", "p1", "p2").write_parquet(os.path.join(WORK, f"test_scores_{model_tag}{sfx}.parquet"))
    ids = s1.select("s", source1_entity_id="s1_id")
    for df, name, col in ((M.group_by("s").agg(pl.col("q").sort()), "matching_results.tsv", "matched_entity_ids"),
                          (C, "candidate_pairs.tsv", "candidate_entity_ids")):
        df = df.with_columns(pl.col("q").list.eval(int_to_id("")).list.join(",").alias(col)).select("s", col)
        out = ids.join(df, on="s", how="left").with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col)
        out.write_csv(os.path.join(OUT, name), separator="\t", quote_style="never")
        log(f"wrote {name}: {out.height} rows, {(out[col] != '').sum()} non-empty")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
