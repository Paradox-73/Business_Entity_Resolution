#!/bin/bash
# Keeps v8 going until its final output exists: if the v8 driver session is gone without summary.json, resume it.
while [ ! -f $HOME/ber/w/v8/summary.json ]; do
  if ! tmux has-session -t v8 2>/dev/null; then
    echo "$(date +%H:%M:%S) v8 driver not running and no final output -> resume" >> $HOME/ber/v8.log
    tmux new-session -d -s v8 "bash $HOME/ber/run_v8_resume.sh 2>&1 | grep --line-buffered -v -i 'warn\|Loading weights' >> $HOME/ber/v8.log"
  fi
  sleep 120
done
