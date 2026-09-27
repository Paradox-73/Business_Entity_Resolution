#!/bin/bash
# usage: save_ce.sh <vN> <output_dir> <held-out score> "<what this version tests>" [extra files to keep...]
# For transformer (reranker) versions: copies the submission file, finalize's result.json and the given extra files
# (rule, stage-3 log) into submissions/<vN>/, appends LOG.md, commits and git-tags <vN>. No data files are committed.
set -e
R="/e/Projects/Amazon ML Challenge"; V="$1"; OD="$2"; SC="$3"; WHAT="$4"; shift 4
mkdir -p "$R/submissions/$V"
cp "$OD/matching_results.tsv" "$R/submissions/$V/matching_results.tsv"
cp "$OD/result.json" "$R/submissions/$V/finalize.json"
for f in "$@"; do cp "$f" "$R/submissions/$V/"; done
N=$(awk -F'\t' 'NR>1 && $2!=""' "$OD/matching_results.tsv" | wc -l)
echo "| $V | \`submissions/$V/\` | $(date '+%a %H:%M') | $SC (held-out) | _upload & fill in_ | $WHAT ($N S1 rows with matches) |" >> "$R/submissions/LOG.md"
cd "$R" && git add -A submissions && git commit -q -m "$V: $WHAT (held-out $SC)" && git tag "$V" && git push -q && git push -q origin "$V"
echo "saved $V: held-out $SC, $N non-empty rows, commit $(git rev-parse --short HEAD)"
