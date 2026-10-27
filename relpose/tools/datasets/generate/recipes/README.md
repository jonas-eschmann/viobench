# datasets/ — recipes for the episode sets

The data itself is not in git (`build/datasets/`). `generate_public.sh` produces the published validation sets and the
real-data sample, `generate_hidden.sh` the server-only sets; both are lists of `relpose simulate` / `relpose convert-euroc`
calls. `docs/DATA.md` describes the episode format, the generator and the conditions; `docs/INSTRUCTOR_GUIDE.md` the
workflow around them.

Both recipes require an absent or empty output directory and check it before generating any episodes. Keep earlier
sets in their own directories: overwriting only some episodes would mix old ten-second flights with the new
one-minute flights. Set `RELPOSE_DATASET_OUTPUT` to choose a fresh destination, for example
`RELPOSE_DATASET_OUTPUT=/data/relpose/public-minute ./relpose/tools/datasets/generate/recipes/generate_public.sh`.
Use a separate destination for the hidden recipe, then point evaluation at the new directories and re-run old
submissions before comparing scores across dataset versions.

EuRoC archives: ETH Research Collection record 20.500.11850/690084 (DSpace API: `/server/api/pid/find?id=20.500.11850/690084`
→ item → ORIGINAL bundle → bitstreams `machine_hall.zip` 11.8 GiB, `vicon_room1.zip` 5.6 GiB, `vicon_room2.zip` 5.6 GiB;
each contains per-sequence `<seq>.zip` in ASL format). Rights: "In Copyright – Non-Commercial Use Permitted". Unzip into
`build/datasets/raw/euroc/<sequence>/mav0/...`.

Reproducibility: HyperDrone output depends on the machine (`-march=native -ffast-math`), so a set is defined by its
files, not its seeds; keep the generated directories (`episode.json` records every parameter and the spawn point).
