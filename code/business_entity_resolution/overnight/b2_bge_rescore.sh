#!/bin/bash
# Friend 2's machine, 27 Sep 10:05: the first ce_b2 bge scores of the new pairs used cut-down test_s* files (texts of
# most new pairs were empty). With the full test files in place, delete those outputs and rescore folds 0-2 (new pairs
# only; old pairs copied from ce_x). Batch 256: the GPU is shared with the e5-large training. Log: work/b2_bge_rescore.log
W=/home/bhavya/Business_Entity_Resolution/work; L=$W/b2_bge_rescore.log
echo 1000 > /proc/self/oom_score_adj
cd /home/bhavya/Business_Entity_Resolution/code/business_entity_resolution/src
for k in 0 1 2; do
  rm -rf $W/ce_b2/parts_test_bgef$k $W/ce_b2/test_ce_bgef$k.parquet
  echo "$(date +%H:%M:%S) fold $k start" >> $L
  BER_CE_DIR=$W/ce_b2 BER_CE_REUSE=$W/ce_x BER_CE_NAME=bge BER_CE_SIDE=f$k BER_CE_MODEL=$W/ce/model_bgef$k BER_CE_SCORE_BS=256 \
    python3 rerank.py score >> $L 2>&1 && echo "$(date +%H:%M:%S) OK r$k" >> $L || { echo "$(date +%H:%M:%S) FAILED r$k" >> $L; exit 1; }
done
echo "$(date +%H:%M:%S) DONE rescore" >> $L
