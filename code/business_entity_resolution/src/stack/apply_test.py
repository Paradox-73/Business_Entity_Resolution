"""Apply the cross-fitted stacker (mean of the models in the file) to the test rows of the countries with training
labels (US, India; common.labelled_countries), movers after the model, decide_expf; write OUT/usi_pairs<tag>.parquet
and, when submissions/v10c exists, a diff against the v10c pairs of those countries.

  python apply_test.py [<tag> [<floor> [<alpha>]]]    models: OUT/lgb_models<tag>.pkl, or the file named by BER_STACK_MODELS
  v10d: python apply_test.py _avg
"""
import os, sys, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/
from common import ROOT  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *
from common import labelled_countries, log
from pipeline import decide_expf
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
FL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
AL = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
models = pickle.load(open(os.environ.get("BER_STACK_MODELS", os.path.join(OUT, f"lgb_models{TAG}.pkl")), "rb"))
feats = models[0].feature_name()
d = pl.read_parquet(os.path.join(OUT, "te.parquet"), columns=["q", "s", "p", "ctry"] + [f for f in feats if f not in ("p", "ctry")])
X = d.select(feats).to_numpy().astype(np.float32)
pst = np.mean([m.predict(X) for m in models.values()], axis=0)
del X
d = d.select("q", "s", "p", "ctry").with_columns(pst=pl.Series(pst.astype(np.float32)))
# movers after the model
ref = i64(pl.read_parquet(os.path.join(BLEND, "test_scores_usi_blend_w04_movers.parquet"))).rename({"p2": "pref"})
rem = pl.read_parquet(os.path.join(MOVERS, "tier12_removed.parquet"), columns=["q", "s"]).with_columns(rm=pl.lit(1))
res = pl.read_parquet(os.path.join(MOVERS, "tier12_restored.parquet"), columns=["q", "s"]).with_columns(rs=pl.lit(1))
d = d.join(ref.select("q", "s"), on=["q", "s"], how="full", coalesce=True).join(rem, on=["q", "s"], how="left").join(res, on=["q", "s"], how="left")
for col in ("p", "pst"):
    d = d.with_columns(pl.when(pl.col("rm") == 1).then(0.0).when(pl.col("rs") == 1).then(pl.max_horizontal(pl.col(col).fill_null(0.0), pl.lit(0.95)))
                       .otherwise(pl.col(col)).cast(pl.Float32).alias(col))
chk = d.join(ref, on=["q", "s"], how="left").filter((pl.col("p") - pl.col("pref")).abs() > 1e-5).height
log(f"test pairs {d.height}; baseline p after movers differing from the blend_second.py p2: {chk}")
base = decide_expf(d.select("q", "s", p2="p"), "p2", 0.5, 1.0)
new = decide_expf(d.select("q", "s", p2="pst"), "p2", FL, AL)
new.write_parquet(os.path.join(OUT, f"usi_pairs{TAG}.parquet"))
log(f"wrote {os.path.join(OUT, f'usi_pairs{TAG}.parquet')}: {new.height} US/India pairs")
# v10c US/India pairs (only where the submission files are present)
V10C = os.path.join(ROOT, "submissions", "v10c", "matching_results.tsv")
if not os.path.exists(V10C):
    sys.exit(0)
s1 = s1_info("test")
v = pl.read_csv(V10C, separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
v = (v.with_columns(q_id=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("q_id").filter(pl.col("q_id") != "")
      .select(s=id_to_int("source1_entity_id").cast(pl.Int64), q=id_to_int("q_id").cast(pl.Int64)))
v = v.join(s1.select("s"), on="s")
out = {}
out["baseline_vs_v10c"] = {"base_pairs": base.height, "v10c_pairs": v.height,
                           "base_minus_v10c": base.join(v, on=["s", "q"], how="anti").height,
                           "v10c_minus_base": v.join(base, on=["s", "q"], how="anti").height}
add = new.join(v, on=["s", "q"], how="anti").join(s1.select("s", "ctry"), on="s")
rmv = v.join(new, on=["s", "q"], how="anti").join(s1.select("s", "ctry"), on="s")
chg = pl.concat([add.select("s"), rmv.select("s")]).unique()
for c, cn in enumerate(labelled_countries()):
    n = s1.filter(pl.col("ctry") == c).height
    out[cn] = {"added": add.filter(pl.col("ctry") == c).height, "removed": rmv.filter(pl.col("ctry") == c).height,
               "rows_changed": chg.join(s1.filter(pl.col("ctry") == c).select("s"), on="s").height, "rows": n}
out["share_rows_changed"] = chg.height / s1.height
out["new_pairs"] = new.height
json.dump(out, open(os.path.join(OUT, f"diff{TAG}.json"), "w"), indent=1)
add.write_parquet(os.path.join(OUT, f"added_vs_v10c{TAG}.parquet"))
rmv.write_parquet(os.path.join(OUT, f"removed_vs_v10c{TAG}.parquet"))
log(json.dumps(out))
