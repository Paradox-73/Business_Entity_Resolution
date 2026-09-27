#!/bin/bash
# After b2_bge.sh (DONE v7q): v7p_hyb = v7q scores (old candidates, small+bge) except records whose wide-build best S1
# is new (hybrid_b2.py, wide = v7p blend); v7q's rule; France from v7ens (v7p_hyb) and v7m (v7p_hyb_frrest). Log: work/b2_hyb.log
set -u
R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"; L="$W/b2_hyb.log"; export PYTHONIOENCODING=utf8
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
log "waiting for DONE v7q"
until grep -q "DONE v7q" "$W/b2_bge.log" 2>/dev/null; do sleep 60; done
cd "$R/code/business_entity_resolution/src"
python hybrid_b2.py "$WW\test_scores_blend_v7q.parquet" "$WW\test_scores_blend_v7p.parquet" "$WW\ce\test_scores_v7p_hyb.parquet" >> "$L" 2>&1 || { log "hybrid FAILED"; exit 1; }
BER_CALIB="$WW\ce\rule_blend_v7q.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\out_v7p_hyb_all" "$WW\ce\test_scores_v7p_hyb.parquet" > "$W/finalize_v7p_hyb.log" 2>&1 || { log "finalize FAILED"; exit 1; }
python splice_country.py "$W/out_v7p_hyb_all/matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$W/out_v7p_hyb" >> "$L" 2>&1
python splice_country.py "$W/out_v7p_hyb_all/matching_results.tsv" "$W/out_v7m/matching_results.tsv" "$W/out_v7p_hyb_frrest" >> "$L" 2>&1
for v in v7p_hyb v7p_hyb_frrest; do log "$v check: $(python check_submission.py "$W/out_$v/matching_results.tsv" 2>&1 | tail -n 1)"; done
log "DONE v7p_hyb"
