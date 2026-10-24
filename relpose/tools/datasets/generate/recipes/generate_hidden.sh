#!/usr/bin/env bash
# The hidden sets (server only): other seeds, HSSD validation scenes (CC BY-NC), other EuRoC sequences. Never publish build/datasets/hidden.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/datasets/hidden
relpose simulate -o $OUT/default       --episodes 32 --seed 90000 --condition default       --scenes hssd-val
relpose simulate -o $OUT/blur          --episodes 16 --seed 91000 --condition blur          --scenes hssd-val
relpose simulate -o $OUT/lowlight      --episodes 16 --seed 92000 --condition lowlight      --scenes hssd-val
relpose simulate -o $OUT/oversaturated --episodes 16 --seed 93000 --condition oversaturated --scenes hssd-val
for sequence in MH_02_easy MH_04_difficult V1_02_medium V2_01_easy V2_02_medium; do
    relpose convert-euroc build/datasets/raw/euroc/$sequence -o $OUT/real --max-snippets 2
done
