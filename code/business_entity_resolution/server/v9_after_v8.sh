#!/bin/bash
# Serialise the two runs on the shared server: start the v9 chain only after v8 has written its final output.
until [ -f $HOME/ber/w/v8/summary.json ]; do sleep 120; done
echo "$(date +%H:%M:%S) v8 finished; starting v9" >> $HOME/ber/v9.log
tmux new-session -d -s v9 "bash $HOME/ber/run_v9.sh 2>&1 | tee -a $HOME/ber/v9.log"
