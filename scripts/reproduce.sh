#!/usr/bin/env bash
# Rerun every analysis reported in README.md and rebuild results/SUMMARY.md.
# Uses the frozen corpora and embeddings in data/, so no network is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

C998=data/legacy_arxiv998/corpus.csv
E998=data/legacy_arxiv998/embeddings_mpnet.npy
RELAXED="--min-balance 0.25 --min-per-field 2 --persistence 0.25"

# Each run() call queues one analysis; they execute JOBS at a time below.
JOBS=${JOBS:-3}
QUEUE=$(mktemp)
run() { echo "$*" >> "$QUEUE"; }

# Full 998-abstract corpus as harvested by v2 (labels by query only).
run --corpus $C998 --embeddings $E998 --subset all --out results/arxiv998_all_mpnet --legacy-check
run --corpus $C998 --embeddings $E998 --subset all --eps-quantile 0.9 --out results/arxiv998_all_mpnet_eps90
run --corpus $C998 --representation lsa --subset all --out results/arxiv998_all_lsa
run --corpus $C998 --representation lsa --subset all --eps-quantile 0.9 --out results/arxiv998_all_lsa_eps90

# Restricted to documents that mention their own field's lexicon.
for rep in mpnet lsa; do
  emb=""; [ "$rep" = mpnet ] && emb="--embeddings $E998"
  for q in 0.5 0.9; do
    sfx=""; [ "$q" = 0.9 ] && sfx="_eps90"
    run --corpus $C998 $emb --representation $rep --subset on_topic --eps-quantile $q \
        --out results/arxiv998_ontopic_${rep}${sfx}
    run --corpus $C998 $emb --representation $rep --subset on_topic --eps-quantile $q $RELAXED \
        --out results/arxiv998_ontopic_${rep}${sfx}_relaxed
  done
done

# 100-abstract pilot.
run --corpus data/legacy_arxiv100/corpus.csv --embeddings data/legacy_arxiv100/embeddings_mpnet.npy \
    --subset all --min-samples 3 --out results/arxiv100_all_mpnet

xargs -P "$JOBS" -I{} sh -c 'echo "== {}"; python scripts/run_analysis.py {} > /dev/null' < "$QUEUE"
rm -f "$QUEUE"

python scripts/summarize_results.py
