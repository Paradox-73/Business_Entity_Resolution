#!/bin/bash
# Waits for bge folds 1 and 2 on friend 2's machine ("OK score fold k" in ce_folds_bge.log) and downloads
# train_ce_bgef{k}/test_ce_bgef{k} into work/ce_x. Log: work/f2_watch.log. Detached so the low-memory reaper can't stop it.
set -u
F2="/c/ber_scratch/f2"; X="E:/Projects/Amazon ML Challenge/work/ce_x"
L="/e/Projects/Amazon ML Challenge/work/f2_watch.log"; RD=/home/bhavya/Business_Entity_Resolution/work
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
cd "$F2"
for k in 1 2; do
  if [ -s "$X/test_ce_bgef$k.parquet" ] && [ -s "$X/train_ce_bgef$k.parquet" ]; then log "fold $k already here"; continue; fi
  log "waiting for OK score fold $k"
  until python f2.py run "grep -c 'OK score fold $k' $RD/ce_folds_bge.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 300; done
  for f in train_ce_bgef$k.parquet test_ce_bgef$k.parquet; do
    rm -f "$X/$f"
    for t in 1 2 3; do MSYS_NO_PATHCONV=1 python f2.py get "$RD/ce_x/$f" "$X/$f" >> "$L" 2>&1 && [ -s "$X/$f" ] && break; sleep 30; done
    log "got $f: $(stat -c %s "$X/$f" 2>/dev/null) bytes"
  done
  log "remote: $(python f2.py run "tail -n 2 $RD/ce_folds_bge.log" 2>&1 | tr '\n' ' ')"
done
log "DONE bge folds"
