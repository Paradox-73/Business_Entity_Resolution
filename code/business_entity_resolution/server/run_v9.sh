#!/bin/bash
# v9: FULL rebuild with the v8 blocking/cleaning (BER_V8=1) -> pipeline.py stages 1-2 (held-out comparable with
# v7ens 0.98813 / v7g 0.98845) -> 3-fold bge-reranker-v2-m3 on close calls -> stage 3 -> France rules -> files.
# Runs next to the v8 build: one heavy build at a time, restarts only with >= 25 GB free. GPU 2 ONLY.
export BER_V8=1 BER_WORK=$HOME/ber/w BER_DATA=$HOME/ber/repo/student_resource/dataset BER_OUT=$HOME/ber/out/v9
export CUDA_VISIBLE_DEVICES=2 BER_THREADS=24 POLARS_MAX_THREADS=24 OMP_NUM_THREADS=24 RAYON_NUM_THREADS=24 PYTHONIOENCODING=utf8
export BER_CHUNK=125000 BER_CE_DIR=$HOME/ber/w/ce_x
PY=$HOME/ber/.venv/bin/python
cd $HOME/ber/repo/code/business_entity_resolution/src
F=$BER_WORK/pairs/full; T=$BER_WORK/pairs/test
avail() { free -g | awk '/Mem:/{print $7}'; }
running() { pgrep -u gathik -f "[p]ipeline.py build $1" > /dev/null; }
traindone() { [ -f $F/India.done ] && [ -f $F/US.done ]; }
testdone() { [ -f $T/France.done ] && [ -f $T/India.done ] && [ -f $T/US.done ]; }
v8testdone() { [ -f $BER_WORK/v8/test_top/US.done ] || grep -q "STEP test DONE\|GIVING UP\|SMOKE FAILED" $HOME/ber/v8.log; }
echo "$(date +%H:%M:%S) v9 start"
while ! (traindone && testdone); do
  if ! traindone && ! running train && [ "$(avail)" -ge 25 ]; then
    echo "$(date +%H:%M:%S) (re)starting train build, available $(avail)GB"
    nohup $PY pipeline.py build train full >> $BER_WORK/build_train.log 2>&1 &
    sleep 120
  fi
  if ! testdone && ! running test && v8testdone && [ "$(avail)" -ge 25 ]; then
    echo "$(date +%H:%M:%S) (re)starting test build, available $(avail)GB"
    nohup $PY pipeline.py build test test >> $BER_WORK/build_test.log 2>&1 &
    sleep 120
  fi
  sleep 60
done
echo "$(date +%H:%M:%S) DONE builds"
step() { for t in 1 2 3; do until [ "$(avail)" -ge 25 ]; do sleep 60; done
  echo "$(date +%H:%M:%S) START $* (try $t)"; "$@" && { echo "$(date +%H:%M:%S) DONE $*"; return 0; }
  echo "$(date +%H:%M:%S) FAILED $* (try $t)"; sleep 60; done; echo "GIVING UP"; exit 1; }
step $PY pipeline.py train full cons
step $PY pipeline.py predict full_xgb_cons test
step $PY rerank.py select full_xgb_cons
M=BAAI/bge-reranker-v2-m3
step $PY ce_folds.py --model $M --name bge --dir $BER_CE_DIR --smoke --no-freeze --batch 32 --lr 2e-5 --score-batch 512
step $PY ce_folds.py --model $M --name bge --dir $BER_CE_DIR --no-freeze --batch 64 --lr 2e-5 --score-batch 1024 --train-rows 1000000
export BER_S3_SIDES=bgef0,bgef1,bgef2
step $PY rerank.py stage3
MD=$BER_WORK/models/full_xgb_cons
cp $BER_CE_DIR/rule_bgef0bgef1bgef2.json $MD/rule_v9.json
step $PY fr_minrule.py $BER_CE_DIR/test_scores_ce_bgef0bgef1bgef2.parquet $BER_WORK/test_scores_full_xgb_cons.parquet $BER_CE_DIR/test_scores_v9_frmin.parquet
BER_CALIB=rule_v9.json BER_FR_LEGAL_VETO=1 step $PY finalize.py full_xgb_cons $BER_OUT $BER_CE_DIR/test_scores_v9_frmin.parquet
step $PY check_submission.py $BER_OUT/matching_results.tsv
echo "$(date +%H:%M:%S) v9 ALL DONE"
