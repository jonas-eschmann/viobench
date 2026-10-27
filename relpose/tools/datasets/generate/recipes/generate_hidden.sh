#!/usr/bin/env bash
# The hidden sets (server only): other seeds, HSSD validation scenes (CC BY-NC), other EuRoC sequences. Never publish build/datasets/hidden.
# One-minute episodes: a submission runs all ~50 of them, so the counts set the evaluation time per submission.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${RELPOSE_DATASET_OUTPUT:-build/datasets/hidden}
shopt -s nullglob dotglob
existing=("$OUT"/*)
if [[ (-e "$OUT" || -L "$OUT") && ! -d "$OUT" ]] || (( ${#existing[@]} )); then
    printf 'error: output must be absent or empty: %s\nChoose a fresh directory with RELPOSE_DATASET_OUTPUT to avoid mixing dataset versions.\n' "$OUT" >&2
    exit 1
fi
relpose simulate -o "$OUT/default"       --episodes 16 --seed 90000 --condition default       --scenes hssd-val
relpose simulate -o "$OUT/blur"          --episodes 8  --seed 91000 --condition blur          --scenes hssd-val
relpose simulate -o "$OUT/lowlight"      --episodes 8  --seed 92000 --condition lowlight      --scenes hssd-val
relpose simulate -o "$OUT/oversaturated" --episodes 8  --seed 93000 --condition oversaturated --scenes hssd-val
for sequence in MH_02_easy MH_04_difficult V1_02_medium V2_01_easy V2_02_medium; do
    relpose convert-euroc build/datasets/raw/euroc/$sequence -o "$OUT/real" --max-snippets 2
done
