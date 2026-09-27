#!/bin/bash
# After e5-large finishes on friend 2 (e5l_after.log: DONE e5l): download folds -> check vs e5-small -> remote scoring of
# the wide new pairs -> stage 3 e5l on ce_x (old candidates) and ce_b2 (wide) -> blends -> v7r (wide: small+bge+e5l)
# and v7s (old candidates: small+bge+e5l), each with France from v7ens / v7m. Log: work/e5l_final.log. Detached.
set -u
F2="/c/ber_scratch/f2"; R="/e/Projects/Amazon ML Challenge"; W="$R/work"; WW="$(cygpath -w "$W")"
RW=/home/bhavya/Business_Entity_Resolution/work; L="$W/e5l_final.log"; export PYTHONIOENCODING=utf8
log() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
f2() { (cd "$F2" && MSYS_NO_PATHCONV=1 python f2.py "$@"); }
log "waiting for e5-large on friend 2"
until f2 run "grep -c 'DONE e5l\|FAILED e5l' $RW/e5l_after.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 300; done
f2 run "grep -c 'DONE e5l' $RW/e5l_after.log" 2>/dev/null | grep -q "^[1-9]" || { log "e5l FAILED on friend 2: $(f2 run "tail -n 2 $RW/e5l_after.log")"; exit 1; }
f2 put "$WW\f2\b2_e5l_remote.sh" "$RW/b2_e5l.sh" >> "$L" 2>&1
f2 run "cd $RW; setsid nohup bash b2_e5l.sh > /dev/null 2>&1 < /dev/null &" >/dev/null 2>&1
log "remote wide scoring started; downloading ce_x folds"
for k in 0 1 2; do for sp in train test; do
  for t in 1 2 3; do f2 get "$RW/ce_x/${sp}_ce_e5lf$k.parquet" "$WW\ce_x\${sp}_ce_e5lf$k.parquet" >> "$L" 2>&1 && break; sleep 30; done
done; done
cd "$R/code/business_entity_resolution/src"
python - >> "$L" 2>&1 <<'PY' || { log "e5l check FAILED"; exit 1; }
import polars as pl
from sklearn.metrics import roc_auc_score
X = "E:/Projects/Amazon ML Challenge/work/ce_x"
for k in range(3):
    a = pl.read_parquet(f"{X}/train_ce_e5lf{k}.parquet"); b = pl.read_parquet(f"{X}/train_ce_bgef{k}.parquet", columns=["q", "s", "ce"]).rename({"ce": "cb"})
    d = a.join(b, on=["q", "s"]); y = d["label"].cast(pl.Int8).to_numpy()
    t = pl.read_parquet(f"{X}/test_ce_e5lf{k}.parquet")
    print(f"e5l f{k}: AUC {roc_auc_score(y, d['ce'].to_numpy()):.5f} (bge {roc_auc_score(y, d['cb'].to_numpy()):.5f}), test rows {t.height}, nulls {t['ce'].null_count()}")
    assert t.height == 3699344 and t["ce"].null_count() == 0
PY
log "$(tail -n 3 "$L" | tr '\n' ' ')"
if [ ! -f "$W/ce_x/test_scores_ce_e5lfolds.parquet" ]; then
  BER_CE_DIR="$WW\ce_x" BER_S3_EXTRA=1 BER_S3_SIDES=e5lf0,e5lf1,e5lf2 BER_S3_TAG=_e5lfolds python rerank.py stage3 > "$W/ce_x_stage3_e5l.log" 2>&1 \
    || { log "stage3 e5l (old) FAILED"; exit 1; }
fi
log "e5l stage 3: $(grep -h 'halves protocol' "$W/ce_x_stage3_e5l.log" | cut -c1-160)"
S_OLD="full_cons:ce_x:_smallfolds full_cons:ce_x:_bgefolds full_cons:ce_x:_e5lfolds"
BER_BLEND_HALVES=1 python blend.py v7s $S_OLD > "$W/blend_v7s_halves.log" 2>&1; log "v7s halves (v7q 0.98878): $(grep -h BEST "$W/blend_v7s_halves.log")"
python blend.py v7s $S_OLD > "$W/blend_v7s.log" 2>&1 || { log "blend v7s FAILED"; exit 1; }
log "v7s all (v7q 0.98950): $(grep -h BEST "$W/blend_v7s.log")"
log "waiting for the remote wide scoring"
until f2 run "grep -c 'DONE e5l wide\|FAILED' $RW/b2_e5l.log" 2>/dev/null | grep -q "^[1-9]"; do sleep 120; done
f2 run "grep -c 'DONE e5l wide' $RW/b2_e5l.log" 2>/dev/null | grep -q "^[1-9]" || { log "wide scoring FAILED: $(f2 run "tail -n 3 $RW/b2_e5l.log")"; exit 1; }
for k in 0 1 2; do
  for t in 1 2 3; do f2 get "$RW/ce_b2/test_ce_e5lf$k.parquet" "$WW\ce_b2\test_ce_e5lf$k.parquet" >> "$L" 2>&1 && break; sleep 30; done
  cp "$W/ce_x/train_ce_e5lf$k.parquet" "$W/ce_b2/"
done
python - >> "$L" 2>&1 <<'PY' || { log "wide e5l check FAILED"; exit 1; }
import polars as pl
W = "E:/Projects/Amazon ML Challenge/work"
for k in range(3):
    a = pl.read_parquet(f"{W}/ce_b2/test_ce_e5lf{k}.parquet"); b = pl.read_parquet(f"{W}/ce_x/test_ce_e5lf{k}.parquet", columns=["q", "s", "ce"]).rename({"ce": "o"})
    j = a.join(b, on=["q", "s"], how="left"); n, o = j.filter(pl.col("o").is_null()), j.filter(pl.col("o").is_not_null())
    print(f"e5l wide f{k}: rows {a.height}, reused max diff {(o['ce'] - o['o']).abs().max()}, new mean {n['ce'].mean():.3f}, old mean {o['ce'].mean():.3f}")
    assert a.height == 3788098 and (o["ce"] - o["o"]).abs().max() == 0
PY
log "$(tail -n 3 "$L" | tr '\n' ' ')"
BER_CE_DIR="$WW\ce_b2" BER_S3_EXTRA=1 BER_S3_SIDES=e5lf0,e5lf1,e5lf2 BER_S3_TAG=_e5lfolds \
  BER_S3_TEST="$WW\test_scores_full_cons_test_b2.parquet" python rerank.py stage3 > "$W/b2_stage3_e5l.log" 2>&1 || { log "stage3 e5l (wide) FAILED"; exit 1; }
python blend.py v7r full_cons:ce_b2:_smallfolds full_cons:ce_b2:_bgefolds full_cons:ce_b2:_e5lfolds > "$W/blend_v7r.log" 2>&1 || { log "blend v7r FAILED"; exit 1; }
log "v7r all (v7p 0.98950): $(grep -h BEST "$W/blend_v7r.log")"
for N in v7r v7s; do
  python fr_minrule.py "$WW\test_scores_blend_$N.parquet" "$WW\test_scores_full_cons.parquet" "$WW\ce\test_scores_blend_${N}_frmin.parquet" > "$W/frmin_$N.log" 2>&1
  BER_CALIB="$WW\ce\rule_blend_$N.json" BER_FR_LEGAL_VETO=1 python finalize.py full_cons "$WW\out_${N}_all" "$WW\ce\test_scores_blend_${N}_frmin.parquet" > "$W/finalize_$N.log" 2>&1 || { log "finalize $N FAILED"; exit 1; }
  log "$N: $(grep -h 'India: rule\|US: rule' "$W/finalize_$N.log" | sed 's/rule.*scored, //' | tr '\n' ' ')"
  python splice_country.py "$WW\out_${N}_all\matching_results.tsv" "$R/submissions/v7ens/matching_results.tsv" "$WW\out_$N" >> "$L" 2>&1
  python splice_country.py "$WW\out_${N}_all\matching_results.tsv" "$WW\out_v7m\matching_results.tsv" "$WW\out_${N}_frrest" >> "$L" 2>&1
  for v in $N ${N}_frrest; do log "$v check: $(python check_submission.py "$WW\out_$v\matching_results.tsv" 2>&1 | tail -n 1)"; done
done
log "DONE e5l"
