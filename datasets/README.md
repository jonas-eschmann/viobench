# Dataset generation (course staff)

The data itself is not in git (`build/datasets/`). These are the commands that produced the published sets and the
pattern for the hidden ones; `docs/DATA.md` describes format and variants.

```bash
# public: validation sets and variant samples (students generate their own training data on the lab GPUs)
python -m eecs116 simulate -o build/datasets/public/default-val   --episodes 16  --seed 2000  --variant default      --scene-names ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23
python -m eecs116 simulate -o build/datasets/public/blur-val      --episodes 8   --seed 2000  --variant blur         --scene-names ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23
python -m eecs116 simulate -o build/datasets/public/lowlight-val  --episodes 8   --seed 2000  --variant lowlight     --scene-names ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23
python -m eecs116 simulate -o build/datasets/public/oversaturated-val --episodes 8 --seed 2000 --variant oversaturated --scene-names ProcTHOR-Train-7 ProcTHOR-Train-11 ProcTHOR-Train-23
python -m eecs116 convert-euroc build/datasets/raw/euroc/V1_01_easy -o build/datasets/public/real-sample --max-snippets 3
# (simulate renders 640x480 RGB and convert-euroc crops to 640x480 grayscale by default; the harness resamples per submission)

# hidden (server only): other seeds, HSSD validation scenes (CC BY-NC), other EuRoC sequences
python -m eecs116 simulate -o build/datasets/hidden/default       --episodes 32 --seed 90000 --variant default       --scenes hssd-val
python -m eecs116 simulate -o build/datasets/hidden/blur          --episodes 16 --seed 91000 --variant blur          --scenes hssd-val
python -m eecs116 simulate -o build/datasets/hidden/lowlight      --episodes 16 --seed 92000 --variant lowlight      --scenes hssd-val
python -m eecs116 simulate -o build/datasets/hidden/oversaturated --episodes 16 --seed 93000 --variant oversaturated --scenes hssd-val
for seq in MH_02_easy MH_04_difficult V1_02_medium V2_01_easy V2_02_medium; do
    python -m eecs116 convert-euroc build/datasets/raw/euroc/$seq -o build/datasets/hidden/real --max-snippets 2
done
```

EuRoC archives come from the ETH Research Collection record 20.500.11850/690084 (DSpace API:
`/server/api/pid/find?id=20.500.11850/690084` → item → ORIGINAL bundle → bitstreams `machine_hall.zip` 11.8 GiB,
`vicon_room1.zip` 5.6 GiB, `vicon_room2.zip` 5.6 GiB; each contains per-sequence `<seq>.zip` (ASL format) and `.bag`).
Rights: "In Copyright – Non-Commercial Use Permitted".

Reproducibility: HyperDrone output depends on the machine (`-march=native -ffast-math`), so a set is defined by its
files, not its seeds; keep the generated directories (and `episode.json`, which records every parameter and the spawn).
Packaging for students: `tar -C build/datasets/public -cf public.tar .` (PNG frames do not compress further).
