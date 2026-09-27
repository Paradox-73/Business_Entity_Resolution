#!/bin/bash
# 27 Sep 10:05: rebuild v7p after the bge rescoring of the new pairs on friend 2 (b2_bge_rescore.sh; the first pass used
# cut-down test files). Download -> sanity check -> stage 3 bge (wide) -> blend small+bge -> France min rule -> finalize
# -> France from v7ens (v7p) / v7m (v7p_frrest) -> hybrid (v7p_hyb). Log: work/b2_final.log. Detached.
set -u
F2="/c/ber_scratch/f2"; R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"
RW=/home/bhavya/Business_Entity_Resolution/work; L="$W/b2_final.log"; export PYTHONIOENCODING=utf8
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
f2() { (cd "$F2" && MSYS_NO_PATHCONV=1 python f2.py "$@"); }
log "waiting for the rescore on friend 2"
until f2 run "grep -c 'DONE rescore\|FAILED' $RW/b2_bge_rescore.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 120; done
f2 run "grep -c 'DONE rescore' $RW/b2_bge_rescore.log" 2>/dev/null | grep -q "^[1-9]" || { log "rescore FAILED: $(f2 run "tail -n 3 $RW/b2_bge_rescore.log")"; exit 1; }
for k in 0 1 2; do
  rm -f "$W/ce_b2/test_ce_bgef$k.parquet"
  for t in 1 2 3; do f2 get "$RW/ce_b2/test_ce_bgef$k.parquet" "$WW\ce_b2\test_ce_bgef$k.parquet" >> "$L" 2>&1 && break; sleep 30; done
done
cd "$R/code/business_entity_resolution/src"
python - >> "$L" 2>&1 <<'PY' || { log "sanity check FAILED"; exit 1; }
import polars as pl
W = "E:/Projects/Amazon ML Challenge/work"
for k in range(3):
    a = pl.read_parquet(f"{W}/ce_b2/test_ce_bgef{k}.parquet")
    b = pl.read_parquet(f"{W}/ce_x/test_ce_bgef{k}.parquet", columns=["q", "s", "ce"]).rename({"ce": "o"})
    j = a.join(b, on=["q", "s"], how="left")
    new, old = j.filter(pl.col("o").is_null()), j.filter(pl.col("o").is_not_null())
    print(f"bge f{k}: rows {a.height}, reused max diff {(old['ce'] - old['o']).abs().max()}, new {new.height} mean {new['ce'].mean():.3f}, old mean {old['ce'].mean():.3f}")
    assert a.height == 3788098 and (old["ce"] - old["o"]).abs().max() == 0 and new["ce"].mean() > -6
PY
log "$(tail -n 3 "$L" | tr '\n' ' ')"
rm -f "$W/ce_b2/test_scores_ce_bgefolds.parquet"
BER_CE_DIR="$WW\ce_b2" BER_S3_EXTRA=1 BER_S3_SIDES=bgef0,bgef1,bgef2 BER_S3_TAG=_bgefolds \
  BER_S3_TEST="$WW\test_scores_full_cons_test_b2.parquet" python rerank.py stage3 > "$W/b2_stage3_bge.log" 2>&1 \
  || { log "stage3 bge FAILED: $(tail -n 3 "$W/b2_stage3_bge.log")"; exit 1; }
log "$(grep -h 'halves protocol' "$W/b2_stage3_bge.log" | cut -c1-160)"
python blend.py v7p full_cons:ce_b2:_smallfolds full_cons:ce_b2:_bgefolds > "$W/blend_v7p.log" 2>&1 || { log "blend FAILED"; exit 1; }
log "v7p all: $(grep -h BEST "$W/blend_v7p.log")"
python fr_minrule.py "$WW\test_scores_blend_v7p.parquet" "$WW\test_scores_full_cons.parquet" "$WW\ce\test_scores_blend_v7p_frmin.parquet" > "$W/frmin_v7p.log" 2>&1
BER_CALIB="$WW\ce\rule_blend_v7p.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\out_v7p_all" \
  "$WW\ce\test_scores_blend_v7p_frmin.parquet" > "$W/finalize_v7p.log" 2>&1 || { log "finalize v7p FAILED"; exit 1; }
log "$(grep -h 'India: rule\|US: rule' "$W/finalize_v7p.log" | sed 's/rule.*scored, //' | tr '\n' ' ')"
python splice_country.py "$WW\out_v7p_all\matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$WW\out_v7p" >> "$L" 2>&1
python splice_country.py "$WW\out_v7p_all\matching_results.tsv" "$WW\out_v7m\matching_results.tsv" "$WW\out_v7p_frrest" >> "$L" 2>&1
python hybrid_b2.py "$WW\test_scores_blend_v7q.parquet" "$WW\test_scores_blend_v7p.parquet" "$WW\ce\test_scores_v7p_hyb.parquet" >> "$L" 2>&1
BER_CALIB="$WW\ce\rule_blend_v7q.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\out_v7p_hyb_all" "$WW\ce\test_scores_v7p_hyb.parquet" > "$W/finalize_v7p_hyb.log" 2>&1
python splice_country.py "$WW\out_v7p_hyb_all\matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$WW\out_v7p_hyb" >> "$L" 2>&1
python splice_country.py "$WW\out_v7p_hyb_all\matching_results.tsv" "$WW\out_v7m\matching_results.tsv" "$WW\out_v7p_hyb_frrest" >> "$L" 2>&1
for v in v7p v7p_frrest v7p_hyb v7p_hyb_frrest; do log "$v check: $(python check_submission.py "$WW\out_$v\matching_results.tsv" 2>&1 | tail -n 1)"; done
log "DONE v7p"
