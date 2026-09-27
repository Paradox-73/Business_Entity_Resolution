#!/bin/bash
# Friend 2's machine: after the bge new-pair scoring (b2_bge_f2.log: OK f2), train + score a second large family:
# multilingual-e5-large, 3 folds, same settings as bge (full fine-tune, batch 64, lr 2e-5). Smoke test first.
# Log: work/e5l_after.log (+ ce_folds_e5l*.log). If RAM runs out the kernel stops this job first (oom_score_adj 1000).
W=/home/bhavya/Business_Entity_Resolution/work; L=$W/e5l_after.log
echo 1000 > /proc/self/oom_score_adj
echo "$(date +%H:%M:%S) waiting for the bge new-pair scoring" >> $L
until grep -q "OK f2\|FAILED f2" $W/b2_bge_f2.log 2>/dev/null; do sleep 60; done
cd /home/bhavya/Business_Entity_Resolution/code/business_entity_resolution/src
echo "$(date +%H:%M:%S) smoke test" >> $L
python3 ce_folds.py --model intfloat/multilingual-e5-large --name e5l --no-freeze --batch 64 --lr 2e-5 --score-batch 1024 --smoke > $W/e5l_smoke_run.log 2>&1 \
  || { echo "$(date +%H:%M:%S) SMOKE FAILED: $(tail -n 3 $W/e5l_smoke_run.log | tr '\n' ' ')" >> $L; exit 1; }
echo "$(date +%H:%M:%S) smoke OK: $(tail -n 2 $W/ce_folds_e5l_smoke.log 2>/dev/null | tr '\n' ' ')" >> $L
python3 ce_folds.py --model intfloat/multilingual-e5-large --name e5l --no-freeze --batch 64 --lr 2e-5 --score-batch 1024 > $W/e5l_full_run.log 2>&1 \
  && echo "$(date +%H:%M:%S) DONE e5l" >> $L || echo "$(date +%H:%M:%S) FAILED e5l: $(tail -n 3 $W/e5l_full_run.log | tr '\n' ' ')" >> $L
