#!/bin/bash
# If free RAM on this shared server gets critical, stop OUR test build (run_after_builds.sh resumes it from finished
# chunks once >= 20 GB is free), so the out-of-memory killer never picks a job of another user.
while true; do
  a=$(free -g | awk '/Mem:/{print $7}')
  if [ "$a" -lt 6 ]; then
    p=$(pgrep -u gathik -f "pipeline.py build test")
    if [ -n "$p" ]; then kill $p; echo "$(date +%H:%M:%S) LOW MEMORY: available ${a}GB -> stopped test build pid $p"; fi
  fi
  if [ $(( $(date +%s) % 900 )) -lt 30 ]; then echo "$(date +%H:%M:%S) mem available ${a}GB"; fi
  sleep 30
done
