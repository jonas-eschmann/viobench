#!/usr/bin/env bash
# The published sets: validation episodes per condition (ProcTHOR training scenes) and three real EuRoC snippets.
# Students generate their own training data with `relpose simulate` on the lab GPUs; nothing pre-rendered is shipped.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/datasets/public
SCENES="ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23"
relpose simulate -o $OUT/default-val       --episodes 16 --seed 2000 --condition default       --scene-names $SCENES
relpose simulate -o $OUT/blur-val          --episodes 8  --seed 2000 --condition blur          --scene-names $SCENES
relpose simulate -o $OUT/lowlight-val      --episodes 8  --seed 2000 --condition lowlight      --scene-names $SCENES
relpose simulate -o $OUT/oversaturated-val --episodes 8  --seed 2000 --condition oversaturated --scene-names $SCENES
relpose convert-euroc build/datasets/raw/euroc/V1_01_easy -o $OUT/real-sample --max-snippets 3
