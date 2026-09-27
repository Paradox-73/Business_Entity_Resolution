#!/bin/bash
# Friend 2's machine: score the wide-search test close calls with e5-large folds 0-2 (new pairs only; old pairs copied
# from ce_x). Needs the FULL test_s* files (in place since 27 Sep 10:00). Log: work/b2_e5l.log
W=/home/bhavya/Business_Entity_Resolution/work; L=$W/b2_e5l.log
echo 1000 > /proc/self/oom_score_adj
cd /home/bhavya/Business_Entity_Resolution/code/business_entity_resolution/src
for k in 0 1 2; do
  ln -sf $W/ce_x/train_ce_e5lf$k.parquet $W/ce_b2/train_ce_e5lf$k.parquet
  BER_CE_DIR=$W/ce_b2 BER_CE_REUSE=$W/ce_x BER_CE_NAME=e5l BER_CE_SIDE=f$k BER_CE_MODEL=$W/ce/model_e5lf$k BER_CE_SCORE_BS=1024 \
    python3 rerank.py score >> $L 2>&1 && echo "$(date +%H:%M:%S) OK e$k" >> $L || { echo "$(date +%H:%M:%S) FAILED e$k" >> $L; exit 1; }
done
echo "$(date +%H:%M:%S) DONE e5l wide" >> $L
