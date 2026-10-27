# Instructor guide

Three jobs: produce the episode sets, run the submission server, keep the toolkit installable. None of them needs a
runtime build: evaluation runs any `.wasm` under wasmtime.

## 1. Setup

```bash
git clone <repo> && cd <repo>
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,data,sim,server]"        # [sim] needs an NVIDIA GPU + CUDA for HyperDrone (first use compiles ~1 min)
pytest                                          # needs a runtime.wasm: relpose/solution/runtime/python/build.sh
```

## 2. Episode sets (`relpose/tools/datasets/generate/recipes`)

`relpose/tools/datasets/generate/recipes/generate_public.sh` writes the published sets into `build/datasets/public/`, `relpose/tools/datasets/generate/recipes/generate_hidden.sh`
the hidden ones into `build/datasets/hidden/` (other seeds, HSSD validation scenes, other EuRoC sequences). Both are
plain lists of `relpose simulate` / `relpose convert-euroc` calls; adjust counts and seeds there.

- Simulated sets: one-minute episodes by default (`--duration`); about 2 min per batch of 8 at 640 × 480 (default
  condition), ~4 min for `blur`; ~360 MB per episode. Each distinct duration or route length JIT-compiles the
  environment once. Frames are spooled to `$RELPOSE_SPOOL` (default: the system temp directory) while a batch renders:
  point it at a fast local disk with a few GB free.
- Real sets: download EuRoC from the ETH Research Collection (record 20.500.11850/690084; `relpose/tools/datasets/generate/recipes/README.md` has the
  API URLs, the archives are 5–12 GiB each), unzip a sequence, run `relpose convert-euroc`.
- Sets are defined by their files, not by seeds (HyperDrone is not bit-reproducible across machines). Keep the
  generated directories; `episode.json` records every parameter and the spawn point.
- Publish `build/datasets/public` as a tarball (PNG frames do not compress further); never publish `hidden/`.

## 3. Server (`relpose/tools/server`)

```bash
mkdir -p server-data && printf "token1 alice\ntoken2 bob\n" > server-data/tokens.txt
RELPOSE_DATASETS=build/datasets/hidden/default:build/datasets/hidden/blur:build/datasets/hidden/lowlight:build/datasets/hidden/oversaturated:build/datasets/hidden/real \
    python -m relpose.tools.server.worker &                       # evaluates queued submissions one at a time
uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000  # upload API + leaderboard
```

Environment: `RELPOSE_SERVER_DIR` (uploads, reports, sqlite; default `./server-data`), `RELPOSE_TOKENS`,
`RELPOSE_MAX_UPLOAD_MIB` (128), `RELPOSE_DAILY_QUOTA` (20), `RELPOSE_EVAL_TIMEOUT` (14400 s). One submission runs
the ~50 one-minute hidden episodes (60 000 frames): a solution at 50 ms/frame takes about 50 minutes, so lower the
quota or the episode counts in `generate_hidden.sh` if the queue backs up. Students use
`curl -F file=@x.wasm -F label=v1 -H "X-Token: ..." /submit`, `/submissions/<id>` (status, summary), `/submissions/<id>/logs`
(their stderr tails), `/leaderboard` (HTML, Pareto front in bold), `/leaderboard.json`.

Hardening for a public deployment: run the worker as a dedicated user in a container without network and with a memory
limit (the guest is sandboxed by wasmtime and capped by `Limits`; the host Python process is not), TLS in front of the
app, rotate tokens, back up `submissions.sqlite` and `reports/`.

## 4. Scoring knobs

- `relpose.tools.runner.sandbox.Limits`: fuel per step (2 × 10¹⁰), init fuel, memory (2 GiB), wall-clock deadlines.
- `relpose.tools.runner.metrics`: ATE without alignment is the headline; RPE at 1 s, rotation RMSE and the Sim(3)
  diagnostic are in every report. `relpose leaderboard` aggregates by dataset category and marks the Pareto front.
- Conditions and their public ranges: `relpose/tools/datasets/generate/conditions.py`; the hidden sets draw from the same ranges.

## 5. Changing the runtime recipe

`relpose/solution/runtime/python/README.md` documents the cross-build (patches, frozen stdlib, reactor link). The ABI constants live
in exactly three places that a test keeps in sync: `relpose/tools/abi/__init__.py`, `relpose/solution/runtime/python/guest/relpose_guest.py`,
`relpose/solution/runtime/native/relpose_abi.h`.
