#!/bin/bash
# Resume the v8 build (steps test, ce, final; each skips finished chunks / folds). Up to 10 tries per step. GPU 2 ONLY.
export BER_V8=1 BER_WORK=$HOME/ber/w BER_DATA=$HOME/ber/repo/student_resource/dataset BER_OUT=$HOME/ber/out
export CUDA_VISIBLE_DEVICES=2 BER_THREADS=24 POLARS_MAX_THREADS=24 OMP_NUM_THREADS=24 RAYON_NUM_THREADS=24 PYTHONIOENCODING=utf8
export BER_CHUNK=125000
PY=$HOME/ber/.venv/bin/python
cd $HOME/ber/repo/code/business_entity_resolution/src
for step in test ce final; do
  for try in $(seq 1 10); do
    until [ "$(free -g | awk '/Mem:/{print $7}')" -ge 20 ]; do sleep 60; done
    echo "$(date +%H:%M:%S) STEP $step resume try $try"
    if $PY experiments/v8.py $step; then echo "$(date +%H:%M:%S) STEP $step DONE"; break; fi
    echo "$(date +%H:%M:%S) STEP $step FAILED (resume try $try)"; sleep 120
  done
done
[ -f $BER_WORK/v8/summary.json ] && echo "$(date +%H:%M:%S) ALL DONE (resume)"
