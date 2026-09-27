"""Rebuild the final submission (v10d, public leaderboard 0.990565) from the saved intermediate files and the pair sets
in ../sets/, then compare both output files with the uploaded ones by md5.

  cd code/business_entity_resolution/src
  python build_final.py

Folders (environment variables; default in brackets):
  BER_WORK   intermediate files of README steps 1-17 [common.WORK]
  BER_SETS   the pair sets and the stacker models file listed in sets/README.md [code/business_entity_resolution/sets]
  BER_BUILD  this script's own intermediate files, 1.5 GB [$BER_WORK/build_final]
  BER_OUT    matching_results.tsv and candidate_pairs.tsv [common.OUT]
  BER_DATA   the dataset; check_submission.py and the official validator read $BER_DATA/test [common.DATA]
  BER_VALIDATOR  the official validate_submission.py [<repo>/student_resource/utils/validate_submission.py]; skipped
             when the file does not exist

Steps, in the order the versions were built (submissions/LOG.md). Steps 2 (finalize.py) to 10 call the scripts that
built those versions; the score edits of steps 1-2 follow README step 16 and france_fix/v9z/recall/r5_build.py and
france_fix/v9z/build/b1_scores.py, and give the same scores as the saved work/frfix3/test_scores_v9y_combo.parquet.
   1  v9y France scores (README step 16): step-10 France scores; desc_veto_set -> p2 = 0, noise_add_set -> p2 =
      max(p2, 0.95), fp_veto_set -> p2 = 0.
   2  v9z France: recall_add_set pairs -> p2 = 0.95 (pairs missing from the score table are appended with p1 = p2 =
      0.95), then moreveto_stem2_set pairs -> p2 = 0; finalize.py with the step-10 France rule and the legal-form veto.
      Only its France rows are used; v9zm and v10a keep them unchanged.
   3  v10a US/India: blend_second.py (0.6 ours + 0.4 the second pipeline, list-mover overrides from tier12_*.parquet,
      decide_expf(0.5, 1.0)); it also writes cand_union.parquet, the candidate list of v10a-v10d.
   4  v10a = step 3 US/India rows + step 2 France rows (assemble_final.py).
   5  v10b = france_recall.py on v10a (+4,825 France pairs from the second pipeline's matches).
   6  v10c = apply_pair_sets.py on v10b: + fr_typo_safe, fr_same_address_safe, fr_amp_safe (+802 France pairs).
   7  v10d US/India: stack/build_test.py (per-pair features of both pipelines) and stack/apply_test.py with
      lgb_models_avg.pkl (mean of 4 LightGBM models, list-mover overrides after the model, decide_expf(0.5, 1.0)).
      Check: its difference from the v10c US/India pairs equals usi_stack_add / usi_stack_remove.
   8  v10d France = apply_pair_sets.py on v10c: - fr_street_veto (-256 France pairs).
   9  assemble_final.py: US/India rows of step 7 + France rows of step 8 -> $BER_OUT/matching_results.tsv.
  10  make_candidates.py: cand_union + the second pipeline's matched pairs + fr_same_address_safe
      -> $BER_OUT/candidate_pairs.tsv.
  11  Checks: check_submission.py, the official validator (--check-ids), md5 of the rebuilt v10a, v10b, v10c and both
      final files against the files uploaded on 27 Sep 2026.

Each step runs in its own process (script or 'python build_final.py step <name>'), so memory is freed between steps.

Inputs from $BER_WORK: cleaned test files (README step 1), France scores (step 10), US/India scores of the wide search
(steps 13-15, and the e5-large family of README section 6, which is feature c of the stacker), the second pipeline's
files (step 17). These come from GPU transformer runs; retraining gives close but not identical scores, so this script
is exact only from the saved score files.
"""
import hashlib
import json
import os
import subprocess
import sys

import polars as pl

import apply_pair_sets
import assemble_final
import france_recall
import make_candidates
from common import DATA, OUT, ROOT, WORK, id_to_int, int_to_id, log

SRC = os.path.dirname(os.path.abspath(__file__))
SETS = os.environ.get("BER_SETS", os.path.join(os.path.dirname(SRC), "sets"))
BUILD = os.environ.get("BER_BUILD", os.path.join(WORK, "build_final"))
VALIDATOR = os.environ.get("BER_VALIDATOR", os.path.join(ROOT, "student_resource", "utils", "validate_submission.py"))
GATHIK_TSV = os.path.join(WORK, "gathik", "v8_matching_results.tsv")                   # README step 17 output
GATHIK_P = os.path.join(WORK, "gathik", "v9", "ce_x_test_scores_v9_frmin.parquet")     # its per-pair probabilities

# md5 of the files uploaded / saved on 27 Sep 2026 (submissions/<version>/)
EXPECTED = {
    "v10a/matching_results.tsv": "c55eb79152a49674df6154c00edcc672",
    "v10b/matching_results.tsv": "76cc451a73f6adb3cf81c9c8f7eab8ba",
    "v10c/matching_results.tsv": "6fc63685e4ef18a00923f165675d0bad",
    "v10d/matching_results.tsv": "18412329111b5d9c43df3a58df0574d1",
    "v10d/candidate_pairs.tsv": "a2ddcb5d26ede94f780fd2c4d87519a7",
}


def st(name):
    """Path of a file in the pair-set folder SETS."""
    return os.path.join(SETS, name)


def bd(*parts):
    """Path inside this script's build folder BUILD."""
    return os.path.join(BUILD, *parts)


def run(script, *args, env=None):
    """Run a pipeline script from src/ in its own process (these scripts do their work at import)."""
    log(f"run {script} {' '.join(args)}")
    subprocess.run([sys.executable, os.path.join(SRC, script), *args], cwd=SRC, check=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf8", **(env or {})})


def md5(path):
    """md5 hex digest of a file, read 4 MB at a time."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mark(b, path):
    """b with a boolean column hit: the (q, s) pair is in the pair set at `path`."""
    return b.join(pl.read_parquet(path, columns=["q", "s"]).with_columns(hit=pl.lit(True)), on=["q", "s"], how="left") \
        .with_columns(pl.col("hit").fill_null(False))


def france_scores_v9y():
    """README step 16: the three France pair sets applied to the step-10 France scores."""
    b = pl.read_parquet(os.path.join(WORK, "ce", "test_scores_blend_ab_a2_frmin.parquet"))
    b = mark(b, st("desc_veto_set.parquet")).with_columns(p2=pl.when("hit").then(0.0).otherwise("p2").cast(pl.Float32)).drop("hit")
    b = mark(b, st("noise_add_set.parquet")).with_columns(
        p2=pl.when("hit").then(pl.max_horizontal("p2", pl.lit(0.95))).otherwise("p2").cast(pl.Float32)).drop("hit")
    b = mark(b, st("fp_veto_set.parquet")).with_columns(p2=pl.when("hit").then(0.0).otherwise("p2").cast(pl.Float32)).drop("hit")
    return b.select("q", "s", "p1", "p2")


def france_scores_v9z(b):
    """v9z: 5,883 recovered France pairs set to p2 = 0.95, then 73 same-stem descriptor-swap pairs set to p2 = 0
    (the rules of the scripts that built work/frfix3/test_scores_v9y_combo.parquet; sets/README.md)."""
    add = pl.read_parquet(st("recall_add_set.parquet"), columns=["q", "s"])
    b = b.join(add.with_columns(a=pl.lit(True)), on=["q", "s"], how="left").with_columns(
        p2=pl.when(pl.col("a").fill_null(False)).then(pl.lit(0.95)).otherwise(pl.col("p2")).cast(pl.Float32)).drop("a")
    new = add.join(b.select("q", "s"), on=["q", "s"], how="anti").select(
        "q", "s", p1=pl.lit(0.95, pl.Float32), p2=pl.lit(0.95, pl.Float32))
    log(f"France recoveries: {add.height} pairs, {add.height - new.height} in the score table, {new.height} appended")
    b = pl.concat([b, new.select(b.columns).cast(b.schema)])
    veto = pl.read_parquet(st("moreveto_stem2_set.parquet"), columns=["q", "s"]).with_columns(z=pl.lit(True))
    b = b.join(veto, on=["q", "s"], how="left", maintain_order="left").with_columns(
        p2=pl.when(pl.col("z").fill_null(False)).then(0.0).otherwise(pl.col("p2")).cast(pl.Float32)).drop("z")
    assert b.select("q", "s").is_duplicated().sum() == 0
    return b


def write_matching(pairs, out_dir):
    """matching_results.tsv from a table of (s, q) integer pairs, in the layout every script of this folder writes."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id"]).select(
        s1_id="entity_id", s=id_to_int("entity_id").cast(pl.Int64))
    P = pairs.select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64))
    df = P.group_by("s").agg(pl.col("q").sort()).with_columns(
        matched_entity_ids=pl.col("q").list.eval(int_to_id("")).list.join(","))
    o = (s1.join(df.select("s", "matched_entity_ids"), on="s", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
           .select(source1_entity_id="s1_id", matched_entity_ids="matched_entity_ids"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "matching_results.tsv")
    o.write_csv(path, separator="\t", quote_style="never")
    return path


def us_india(tsv):
    """US/India (s, q) pairs of a matching_results.tsv."""
    s1 = pl.read_parquet(os.path.join(WORK, "test_s1.parquet"), columns=["entity_id", "country"]).select(
        s=id_to_int("entity_id").cast(pl.Int64), country="country")
    return apply_pair_sets.pairs(tsv).join(s1, on="s").filter(pl.col("country") != "France").select("s", "q")


def same_pairs(a, b):
    """True if the two (s, q) tables hold the same pairs."""
    return a.height == b.height and a.join(b, on=["s", "q"], how="anti").height == 0


def step_v9z_scores():
    """Steps 1-2 (score file; finalize.py runs next in its own process)."""
    france_scores_v9z(france_scores_v9y()).write_parquet(bd("test_scores_v9z_france.parquet"))


def step_v10a():
    """Step 4: v10a = US/India matches of the blend (step 3) + France rows of v9z (step 2)."""
    ui = write_matching(pl.read_parquet(bd("blend", "usi_pairs.parquet")), bd("v10a_usi"))
    assemble_final.main(ui, bd("v9z_all", "matching_results.tsv"), bd("v10a"))


def step_v10b():
    """Step 5: v10b = france_recall.py on v10a."""
    france_recall.main(bd("v10a", "matching_results.tsv"), GATHIK_TSV, GATHIK_P, bd("v10b"))


def step_v10c():
    """Step 6: v10c = v10b + the three France pair sets."""
    apply_pair_sets.main(bd("v10b", "matching_results.tsv"), bd("v10c"),
                         "+" + st("fr_typo_safe.parquet"), "+" + st("fr_same_address_safe.parquet"), "+" + st("fr_amp_safe.parquet"))


def step_v10d():
    """Steps 7 (check), 8, 9 and 10."""
    usi = pl.read_parquet(bd("stack", "usi_pairs_avg.parquet")).select(pl.col("s").cast(pl.Int64), pl.col("q").cast(pl.Int64))
    v10c = us_india(bd("v10c", "matching_results.tsv"))
    ok = {"usi_stack_add": same_pairs(usi.join(v10c, on=["s", "q"], how="anti"),
                                      pl.read_parquet(st("usi_stack_add.parquet"), columns=["s", "q"])),
          "usi_stack_remove": same_pairs(v10c.join(usi, on=["s", "q"], how="anti"),
                                         pl.read_parquet(st("usi_stack_remove.parquet"), columns=["s", "q"]))}
    log(f"stacker US/India pairs {usi.height}; its difference from v10c equals the saved sets: {ok}")
    json.dump(ok, open(bd("stack_check.json"), "w"))
    apply_pair_sets.main(bd("v10c", "matching_results.tsv"), bd("v10d_france"), "-" + st("fr_street_veto.parquet"))
    ui = write_matching(usi, bd("v10d_usi"))
    assemble_final.main(ui, bd("v10d_france", "matching_results.tsv"), OUT)
    # every pair either pipeline scored + the matched pairs of the second pipeline and of the same-address rule
    make_candidates.main(bd("blend", "cand_union.parquet"), os.path.join(OUT, "matching_results.tsv"), OUT,
                         GATHIK_TSV, st("fr_same_address_safe.parquet"))


STEPS = {f.__name__[5:]: f for f in (step_v9z_scores, step_v10a, step_v10b, step_v10c, step_v10d)}


def main():
    """Every step runs in its own process, so that no process holds more than one step's tables (peak ~5.5 GB)."""
    os.makedirs(BUILD, exist_ok=True)
    log(f"work {WORK}; sets {SETS}; build {BUILD}; out {OUT}")
    self = os.path.basename(__file__)
    run(self, "step", "v9z_scores")                                                          # 1-2
    run("finalize.py", "full_cons", bd("v9z_all"), bd("test_scores_v9z_france.parquet"),
        env={"BER_CALIB": os.path.join(WORK, "ce", "rule_blend_ab_a2.json"), "BER_FR_LEGAL_VETO": "1"})
    ref = os.path.join(WORK, "blend9")                                                       # 3: compared when present
    run("blend_second.py", bd("blend"), SETS, *([ref] if os.path.isdir(ref) else []))
    for name in ("v10a", "v10b", "v10c"):                                                    # 4-6
        run(self, "step", name)
    senv = {"BER_STACK_DIR": bd("stack"), "BER_MOVERS": SETS, "BER_BLEND": bd("blend"),      # 7
            "BER_STACK_MODELS": st("lgb_models_avg.pkl")}
    run(os.path.join("stack", "build_test.py"), env=senv)
    run(os.path.join("stack", "apply_test.py"), "_avg", env=senv)
    run(self, "step", "v10d")                                                                # 7 check, 8-10

    # 11. checks
    m, c = os.path.join(OUT, "matching_results.tsv"), os.path.join(OUT, "candidate_pairs.tsv")
    run("check_submission.py", m)
    if os.path.exists(VALIDATOR):
        log(f"run {VALIDATOR}")
        subprocess.run([sys.executable, VALIDATOR, "-m", m, "-c", c, "-t", os.path.join(DATA, "test"), "--check-ids"],
                       check=True, env={**os.environ, "PYTHONIOENCODING": "utf8"})
    else:
        log(f"official validator not found ({VALIDATOR}); skipped")
    files = {"v10a/matching_results.tsv": bd("v10a", "matching_results.tsv"),
             "v10b/matching_results.tsv": bd("v10b", "matching_results.tsv"),
             "v10c/matching_results.tsv": bd("v10c", "matching_results.tsv"),
             "v10d/matching_results.tsv": m, "v10d/candidate_pairs.tsv": c}
    print(f"{'file':28s} {'rebuilt md5':34s} {'uploaded md5':34s} same")
    bad = 0
    for k, p in files.items():
        h = md5(p)
        bad += h != EXPECTED[k]
        print(f"{k:28s} {h:34s} {EXPECTED[k]:34s} {h == EXPECTED[k]}")
    ok = json.load(open(bd("stack_check.json")))
    bad += sum(not v for v in ok.values())
    print("ALL IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "step":   # one step, run by main() in a child process
        STEPS[sys.argv[2]]()
    else:
        main()
