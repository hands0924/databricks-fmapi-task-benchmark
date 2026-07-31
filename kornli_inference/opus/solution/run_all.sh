#!/usr/bin/env bash
# Full reproducible pipeline. Run from the task directory:
#   bash solution/run_all.sh
set -euo pipefail
mkdir -p cache outputs
python -u solution/make_cache.py    # sparse lexical views + dense engineered features
python -u solution/make_lsa.py 200  # LSA sentence embeddings -> pair interactions
python -u solution/make_align.py    # soft token-alignment features
python -u solution/run_final.py     # 5-fold OOF + full-fold-bagged test probabilities
python -u solution/blend_submit.py  # blend + premise-group constraint -> submission
