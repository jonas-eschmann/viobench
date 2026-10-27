#!/usr/bin/env bash
# The published sets: validation episodes per condition (ProcTHOR training scenes) and two real EuRoC snippets.
# Episodes are one minute long (the `relpose simulate` / `convert-euroc` defaults; ~360 MB each at 640 x 480), so the
# counts are kept small: ~7 GB in total. Students generate their own training data with `relpose simulate` on the lab
# GPUs; nothing pre-rendered is shipped.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${RELPOSE_DATASET_OUTPUT:-build/datasets/public}
shopt -s nullglob dotglob
existing=("$OUT"/*)
if [[ (-e "$OUT" || -L "$OUT") && ! -d "$OUT" ]] || (( ${#existing[@]} )); then
    printf 'error: output must be absent or empty: %s\nChoose a fresh directory with RELPOSE_DATASET_OUTPUT to avoid mixing dataset versions.\n' "$OUT" >&2
    exit 1
fi
SCENES="ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23"
relpose simulate -o "$OUT/default-val"       --episodes 8 --seed 2000 --condition default       --scene-names $SCENES
relpose simulate -o "$OUT/blur-val"          --episodes 4 --seed 2000 --condition blur          --scene-names $SCENES
relpose simulate -o "$OUT/lowlight-val"      --episodes 4 --seed 2000 --condition lowlight      --scene-names $SCENES
relpose simulate -o "$OUT/oversaturated-val" --episodes 4 --seed 2000 --condition oversaturated --scene-names $SCENES
relpose convert-euroc build/datasets/raw/euroc/V1_01_easy -o "$OUT/real-sample" --max-snippets 2
