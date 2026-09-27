#!/bin/bash
# v8 overnight build (v8.py): smoke test, then the full run step by step with retries. GPU 2 ONLY.
export BER_V8=1 BER_WORK=$HOME/ber/w BER_DATA=$HOME/ber/repo/student_resource/dataset BER_OUT=$HOME/ber/out
export CUDA_VISIBLE_DEVICES=2 BER_THREADS=24 POLARS_MAX_THREADS=24 OMP_NUM_THREADS=24 RAYON_NUM_THREADS=24 PYTHONIOENCODING=utf8
export BER_CHUNK=125000
PY=$HOME/ber/.venv/bin/python
cd $HOME/ber/repo/code/business_entity_resolution/src
echo "$(date +%H:%M:%S) SMOKE start"
BER_V8_SMOKE=1 BER_V8_TAG=_smoke $PY v8.py all || { echo "$(date +%H:%M:%S) SMOKE FAILED"; exit 1; }
echo "$(date +%H:%M:%S) SMOKE ok"
for step in train gbdt test ce final; do
  for try in 1 2 3; do
    until [ "$(free -g | awk '/Mem:/{print $7}')" -ge 20 ]; do sleep 60; done
    echo "$(date +%H:%M:%S) STEP $step try $try"
    if $PY v8.py $step; then echo "$(date +%H:%M:%S) STEP $step DONE"; break; fi
    echo "$(date +%H:%M:%S) STEP $step FAILED (try $try)"
    [ $try = 3 ] && { echo "GIVING UP"; exit 1; }
    sleep 60
  done
done
echo "$(date +%H:%M:%S) ALL DONE"
