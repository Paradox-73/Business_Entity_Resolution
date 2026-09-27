#!/bin/bash
# After blocking_b2.py build: stage 1/2 on the wide test candidates -> close calls -> e5-small 3 folds (new pairs only)
# -> stage 3 -> US/India file v7n; France rows spliced from v7ens (v7n) and from v7m (v7n_frrest).
# Log: work/b2_chain.log. Resumable (each step skips finished output).
set -u
R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"; L="$W/b2_chain.log"
cd "$R/code/business_entity_resolution/src"; export PYTHONIOENCODING=utf8
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
log "waiting for the build"
until [ -f "$W/pairs/test_b2/India.done" ] && [ -f "$W/pairs/test_b2/US.done" ]; do sleep 60; done
sleep 30
log "build done: $(ls "$W/pairs/test_b2" | grep -c parquet) files"
if [ ! -f "$W/test_scores_full_cons_test_b2.parquet" ]; then
  python blocking_b2.py predict >> "$W/b2_predict.log" 2>&1 || { log "predict FAILED: $(tail -n 3 "$W/b2_predict.log")"; exit 1; }
fi
log "$(grep 'check\|wrote' "$W/b2_predict.log" | tail -4 | tr '\n' ' ')"
if [ ! -f "$W/ce_b2/test_rows.parquet" ]; then
  python blocking_b2.py rows >> "$W/b2_rows.log" 2>&1 || { log "rows FAILED: $(tail -n 3 "$W/b2_rows.log")"; exit 1; }
fi
log "$(grep 'test rows' "$W/b2_rows.log" | tail -1)"
for k in 0 1 2; do
  [ -f "$W/ce_b2/test_ce_smallf$k.parquet" ] && continue
  BER_CE_DIR="$WW\\ce_b2" BER_CE_SIDE=f$k BER_CE_NAME=small BER_CE_REUSE="$WW\\ce_x" python rerank.py score >> "$W/b2_score_small.log" 2>&1 \
    || { log "score small f$k FAILED: $(tail -n 3 "$W/b2_score_small.log")"; exit 1; }
  log "small f$k scored: $(grep 'reused\|close-call rows' "$W/b2_score_small.log" | tail -2 | tr '\n' ' ')"
done
if [ ! -f "$W/ce_b2/test_scores_ce_smallfolds.parquet" ]; then
  BER_CE_DIR="$WW\\ce_b2" BER_S3_EXTRA=1 BER_S3_SIDES=smallf0,smallf1,smallf2 BER_S3_TAG=_smallfolds \
    BER_S3_TEST="$WW\\test_scores_full_cons_test_b2.parquet" python rerank.py stage3 > "$W/b2_stage3_small.log" 2>&1 \
    || { log "stage3 FAILED: $(tail -n 3 "$W/b2_stage3_small.log")"; exit 1; }
fi
log "$(grep -h 'halves protocol\|wrote' "$W/b2_stage3_small.log" | cut -c1-220 | tr '\n' ' ')"
BER_CALIB="$WW\\ce_b2\\rule_smallfolds.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\\out_v7n_all" \
  "$WW\\ce_b2\\test_scores_ce_smallfolds.parquet" > "$W/b2_finalize.log" 2>&1 || { log "finalize FAILED"; exit 1; }
python experiments/splice_country.py "$W/out_v7n_all/matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$W/out_v7n" >> "$L" 2>&1
python experiments/splice_country.py "$W/out_v7n_all/matching_results.tsv" "$W/out_v7m/matching_results.tsv" "$W/out_v7n_frrest" >> "$L" 2>&1
for v in v7n v7n_frrest; do log "$v check: $(python check_submission.py "$W/out_$v/matching_results.tsv" 2>&1 | tail -n 1)"; done
log "DONE v7n"
