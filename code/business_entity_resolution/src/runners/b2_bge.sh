#!/bin/bash
# After b2_chain.sh has made work/ce_b2 (wide-search test close calls): score its new pairs with bge folds 0-2 on
# friend 2's GPU (f2/b2_bge_remote.sh; each fold once its model is done there), then
#   v7p  = wide search + stage 3 of e5-small and bge (3 folds each) blended; France from v7ens / v7m (_frrest)
#   v7q  = old candidates + e5-small and bge blended (build_combo.sh); France from v7ens / v7m (_frrest)
# Log: work/b2_bge.log. Resumable. Detached (PowerShell Start-Process) so closing the terminal can't stop it.
set -u
F2="${BER_SCRATCH:-/c/ber_scratch}/f2"; R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"
RW=/home/bhavya/Business_Entity_Resolution/work; L="$W/b2_bge.log"; export PYTHONIOENCODING=utf8 MSYS_NO_PATHCONV=1
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
f2() { (cd "$F2" && python f2.py "$@"); }
waitmem() { while [ "$(powershell -NoProfile -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")" -lt 5 ]; do sleep 30; done; }
log "waiting for work/ce_b2/test_rows.parquet"
until grep -q "test rows" "$W/b2_chain.log" 2>/dev/null && [ -s "$W/ce_b2/test_rows.parquet" ]; do sleep 60; done
f2 run "mkdir -p $RW/ce_b2" >/dev/null
f2 put "$WW\ce_b2\test_rows.parquet" "$RW/ce_b2/test_rows.parquet" >> "$L" 2>&1 || { log "upload FAILED"; exit 1; }
f2 put "$WW\f2\b2_bge_remote.sh" "$RW/b2_bge.sh" >> "$L" 2>&1
for k in 0 1 2; do
  [ -s "$W/ce_b2/test_ce_bgef$k.parquet" ] && continue
  # the fold jobs hold ~40 GB of the GPU while scoring (smoke test 04:40 ran out of GPU memory): start only after fold 2
  log "fold $k: waiting for all bge folds to finish on friend 2's machine"
  until f2 run "grep -c 'OK score fold 2' $RW/ce_folds_bge.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 300; done
  if ! f2 run "grep -q 'OK f$k' $RW/b2_bge_f$k.log 2>/dev/null && echo yes" 2>/dev/null | grep -q yes; then
    f2 run "rm -f $RW/b2_bge_f$k.log; setsid nohup bash $RW/b2_bge.sh $k > /dev/null 2>&1 < /dev/null &" >/dev/null 2>&1
    log "fold $k: scoring new pairs started"
    until f2 run "grep -c 'OK f$k\|FAILED f$k' $RW/b2_bge_f$k.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 120; done
  fi
  log "fold $k: $(f2 run "grep 'reused\|close-call rows\|OK\|FAILED\|Error' $RW/b2_bge_f$k.log | tail -4" 2>&1 | tr '\n' ' ' | cut -c1-400)"
  f2 run "grep -c 'OK f$k' $RW/b2_bge_f$k.log" 2>/dev/null | grep -q "^[1-9]" || { log "fold $k FAILED"; exit 1; }   # f2.py exits 0 either way
  for t in 1 2 3; do f2 get "$RW/ce_b2/test_ce_bgef$k.parquet" "$WW\ce_b2\test_ce_bgef$k.parquet" >> "$L" 2>&1 && break; sleep 30; done
done
log "waiting for the train files in ce_x (f2_watch.sh) and for b2_chain.sh"
until [ -s "$W/ce_x/train_ce_bgef1.parquet" ] && [ -s "$W/ce_x/train_ce_bgef2.parquet" ] && [ -s "$W/ce_x/test_ce_bgef2.parquet" ] \
      && grep -q "DONE v7n" "$W/b2_chain.log"; do sleep 60; done
sleep 60
for k in 0 1 2; do [ -f "$W/ce_b2/train_ce_bgef$k.parquet" ] || cp "$W/ce_x/train_ce_bgef$k.parquet" "$W/ce_b2/"; done
cd "$R/code/business_entity_resolution/src"
if [ ! -f "$W/ce_b2/test_scores_ce_bgefolds.parquet" ]; then
  waitmem; log "stage 3 bge (wide candidates)"
  BER_CE_DIR="$WW\ce_b2" BER_S3_EXTRA=1 BER_S3_SIDES=bgef0,bgef1,bgef2 BER_S3_TAG=_bgefolds \
    BER_S3_TEST="$WW\test_scores_full_cons_test_b2.parquet" python rerank.py stage3 > "$W/b2_stage3_bge.log" 2>&1 \
    || { log "stage3 bge FAILED: $(tail -n 3 "$W/b2_stage3_bge.log")"; exit 1; }
fi
log "$(grep -h 'halves protocol\|BEST eval' "$W/b2_stage3_bge.log" | cut -c1-220 | tr '\n' ' ')"
SPECS="full_cons:ce_b2:_smallfolds full_cons:ce_b2:_bgefolds"
waitmem; BER_BLEND_HALVES=1 python blend.py v7p $SPECS > "$W/blend_v7p_halves.log" 2>&1
log "v7p halves (v7ens 0.98813, v7g 0.98845): $(grep -h BEST "$W/blend_v7p_halves.log")"
waitmem; python blend.py v7p $SPECS > "$W/blend_v7p.log" 2>&1 || { log "blend FAILED"; exit 1; }
log "v7p all: $(grep -h BEST "$W/blend_v7p.log")"
python fr_minrule.py "$WW\test_scores_blend_v7p.parquet" "$WW\test_scores_full_cons.parquet" "$WW\ce\test_scores_blend_v7p_frmin.parquet" > "$W/frmin_v7p.log" 2>&1
BER_CALIB="$WW\ce\rule_blend_v7p.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\out_v7p_all" \
  "$WW\ce\test_scores_blend_v7p_frmin.parquet" > "$W/finalize_v7p.log" 2>&1 || { log "finalize v7p FAILED"; exit 1; }
python experiments/splice_country.py "$W/out_v7p_all/matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$W/out_v7p" >> "$L" 2>&1
python experiments/splice_country.py "$W/out_v7p_all/matching_results.tsv" "$W/out_v7m/matching_results.tsv" "$W/out_v7p_frrest" >> "$L" 2>&1
for v in v7p v7p_frrest; do log "$v check: $(python check_submission.py "$W/out_$v/matching_results.tsv" 2>&1 | tail -n 1)"; done
log "DONE v7p"
waitmem; log "v7q: old candidates, e5-small + bge"
bash "$R/code/business_entity_resolution/src/runners/build_combo.sh" v7q small bge >> "$L" 2>&1
python experiments/splice_country.py "$W/out_v7q/matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$W/out_v7q_fr7ens" >> "$L" 2>&1
python experiments/splice_country.py "$W/out_v7q/matching_results.tsv" "$W/out_v7m/matching_results.tsv" "$W/out_v7q_frrest" >> "$L" 2>&1
for v in v7q_fr7ens v7q_frrest; do log "$v check: $(python check_submission.py "$W/out_$v/matching_results.tsv" 2>&1 | tail -n 1)"; done
log "DONE v7q"
