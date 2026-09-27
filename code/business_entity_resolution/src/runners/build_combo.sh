#!/bin/bash
# Build a submission from 3-fold transformer families (files work/ce_x/{train,test}_ce_<fam>f{0,1,2}.parquet).
#   bash code/business_entity_resolution/src/runners/build_combo.sh <name> <fam> [<fam> ...] [+old]
#   e.g. bash code/business_entity_resolution/src/runners/build_combo.sh v7g small base          (+old also blends v7ens's transformers A+B and A2)
# Steps: stage 3 per family (held-out, both protocols) -> blend (held-out halves protocol = comparable with v7ens
# 0.98813; then the normal protocol, which writes the test file and rule) -> France: transformer may only lower
# (fr_minrule) -> two files: <name> (as v7ens's France) and <name>_num (+ France house-number veto, v7k's rule).
# Logs: work/combo_<name>.log. One job at a time; each step waits for >= 5 GB free RAM.
set -u
R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"; L="$W/combo_$1.log"
cd "$R/code/business_entity_resolution/src"; export PYTHONIOENCODING=utf8
NAME="$1"; shift
OLD=""; FAMS=()
for a in "$@"; do if [ "$a" = "+old" ]; then OLD="full_cons:ce:_ab full_cons:ce:_a2"; else FAMS+=("$a"); fi; done
waitmem() { while [ "$(powershell -NoProfile -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")" -lt 5 ]; do sleep 20; done; }
log() { echo "$(date '+%H:%M') $*" | tee -a "$L"; }
SPECS=""
for F in "${FAMS[@]}"; do
  for k in 0 1 2; do for sp in train test; do
    [ -f "$W/ce_x/${sp}_ce_${F}f$k.parquet" ] || { log "MISSING $W/ce_x/${sp}_ce_${F}f$k.parquet"; exit 1; }
  done; done
  if [ ! -f "$W/ce_x/oof_s3_${F}folds.parquet" ]; then
    waitmem; log "stage 3 family $F"
    BER_CE_DIR="$WW\\ce_x" BER_S3_EXTRA=1 BER_S3_SIDES=${F}f0,${F}f1,${F}f2 BER_S3_TAG=_${F}folds \
      python rerank.py stage3 > "$W/ce_x_stage3_$F.log" 2>&1 || { log "stage 3 $F FAILED"; tail -n 5 "$W/ce_x_stage3_$F.log"; exit 1; }
  fi
  log "family $F: $(grep -h 'halves protocol\|BEST eval' "$W/ce_x_stage3_$F.log" | tr '\n' ' ' | cut -c1-400)"
  SPECS="$SPECS full_cons:ce_x:_${F}folds"
done
SPECS="$SPECS $OLD"
waitmem; log "blend (halves protocol, comparable with v7ens 0.98813): $SPECS"
BER_BLEND_HALVES=1 python blend.py "$NAME" $SPECS > "$W/blend_${NAME}_halves.log" 2>&1
log "$(grep -h 'BEST' "$W/blend_${NAME}_halves.log")"
waitmem; log "blend (all close calls rescored, writes the test file)"
python blend.py "$NAME" $SPECS > "$W/blend_$NAME.log" 2>&1 || { log "blend FAILED"; exit 1; }
log "$(grep -h 'BEST' "$W/blend_$NAME.log")"
waitmem; python fr_minrule.py "$WW\\test_scores_blend_$NAME.parquet" "$WW\\test_scores_full_cons.parquet" "$WW\\ce\\test_scores_blend_${NAME}_frmin.parquet" > "$W/frmin_$NAME.log" 2>&1
python experiments/fr_num.py "$WW\\ce\\test_scores_blend_${NAME}_frmin.parquet" "$WW\\ce\\test_scores_blend_${NAME}_frnum.parquet" > "$W/frnum_$NAME.log" 2>&1
for V in "" "_num"; do
  S="$WW\\ce\\test_scores_blend_${NAME}_frmin.parquet"; [ "$V" = "_num" ] && S="$WW\\ce\\test_scores_blend_${NAME}_frnum.parquet"
  BER_CALIB="$WW\\ce\\rule_blend_$NAME.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\\out_$NAME$V" "$S" > "$W/finalize_$NAME$V.log" 2>&1
  log "out_$NAME$V: $(grep -h 'France:' "$W/finalize_$NAME$V.log" | cut -c1-160) | check: $(python check_submission.py "$WW\\out_$NAME$V\\matching_results.tsv" 2>&1 | tail -n 1)"
done
log "DONE $NAME"
