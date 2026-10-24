# Instructor guide

Three jobs: produce the episode sets, run the submission server, keep the toolkit installable. None of them needs a
runtime build: evaluation runs any `.wasm` under wasmtime.

## 1. Setup

```bash
git clone <repo> && cd <repo>
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,data,sim,server]"        # [sim] needs an NVIDIA GPU + CUDA for HyperDrone (first use compiles ~1 min)
pytest                                          # needs a runtime.wasm: submission/toolchain/build-image.sh && submission/python/build.sh
```

## 2. Episode sets (`datasets/`)

`datasets/generate_public.sh` writes the published sets into `build/datasets/public/`, `datasets/generate_hidden.sh`
the hidden ones into `build/datasets/hidden/` (other seeds, HSSD validation scenes, other EuRoC sequences). Both are
plain lists of `relpose simulate` / `relpose convert-euroc` calls; adjust counts and seeds there.

- Simulated sets: ~20 s per batch of 8 episodes at 640 × 480 (default condition), ~45 s for `blur`; ~60 MB per episode.
- Real sets: download EuRoC from the ETH Research Collection (record 20.500.11850/690084; `datasets/README.md` has the
  API URLs, the archives are 5–12 GiB each), unzip a sequence, run `relpose convert-euroc`.
- Sets are defined by their files, not by seeds (HyperDrone is not bit-reproducible across machines). Keep the
  generated directories; `episode.json` records every parameter and the spawn point.
- Publish `build/datasets/public` as a tarball (PNG frames do not compress further); never publish `hidden/`.

## 3. Server (`relpose/server`)

```bash
mkdir -p server-data && printf "token1 alice\ntoken2 bob\n" > server-data/tokens.txt
RELPOSE_DATASETS=build/datasets/hidden/default:build/datasets/hidden/blur:build/datasets/hidden/lowlight:build/datasets/hidden/oversaturated:build/datasets/hidden/real \
    python -m relpose.server.worker &                       # evaluates queued submissions one at a time
uvicorn relpose.server.app:app --host 0.0.0.0 --port 8000  # upload API + leaderboard
```

Environment: `RELPOSE_SERVER_DIR` (uploads, reports, sqlite; default `./server-data`), `RELPOSE_TOKENS`,
`RELPOSE_MAX_UPLOAD_MIB` (128), `RELPOSE_DAILY_QUOTA` (20), `RELPOSE_EVAL_TIMEOUT` (7200 s). Students use
`curl -F file=@x.wasm -F label=v1 -H "X-Token: ..." /submit`, `/submissions/<id>` (status, summary), `/submissions/<id>/logs`
(their stderr tails), `/leaderboard` (HTML, Pareto front in bold), `/leaderboard.json`.

Hardening for a public deployment: run the worker as a dedicated user in a container without network and with a memory
limit (the guest is sandboxed by wasmtime and capped by `Limits`; the host Python process is not), TLS in front of the
app, rotate tokens, back up `submissions.sqlite` and `reports/`.

## 4. Scoring knobs

- `relpose.evaluation.sandbox.Limits`: fuel per step (2 × 10¹⁰), init fuel, memory (2 GiB), wall-clock deadlines.
- `relpose.evaluation.metrics`: ATE without alignment is the headline; RPE at 1 s, rotation RMSE and the Sim(3)
  diagnostic are in every report. `relpose leaderboard` aggregates by dataset category and marks the Pareto front.
- Conditions and their public ranges: `relpose/datasets/conditions.py`; the hidden sets draw from the same ranges.

## 5. Changing the runtime recipe

`submission/python/README.md` documents the cross-build (patches, frozen stdlib, reactor link). The ABI constants live
in exactly three places that a test keeps in sync: `relpose/abi/__init__.py`, `submission/python/guest/relpose_guest.py`,
`submission/native/relpose_abi.h`.
