#!/bin/bash
# Friend 2's machine: score the wide-search test close calls (work/ce_b2/test_rows.parquet) with bge fold k.
# Pairs already scored in work/ce_x are copied, so only the new pairs go through the GPU. Log: work/b2_bge_f<k>.log
k=$1
W=/home/bhavya/Business_Entity_Resolution/work
cd /home/bhavya/Business_Entity_Resolution/code/business_entity_resolution/src
echo 1000 > /proc/self/oom_score_adj     # if RAM runs out, the kernel stops this job first, never someone else's
ln -sf $W/ce_x/train_ce_bgef$k.parquet $W/ce_b2/train_ce_bgef$k.parquet    # train rows are unchanged: kept as is
BER_CE_DIR=$W/ce_b2 BER_CE_REUSE=$W/ce_x BER_CE_NAME=bge BER_CE_SIDE=f$k BER_CE_MODEL=$W/ce/model_bgef$k BER_CE_SCORE_BS=1024 \
  python3 rerank.py score >> $W/b2_bge_f$k.log 2>&1 && echo "OK f$k" >> $W/b2_bge_f$k.log || echo "FAILED f$k" >> $W/b2_bge_f$k.log
