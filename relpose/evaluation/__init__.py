"""Evaluation: solution.wasm → numbers.

- `engine`: the wasmtime engine (fuel metering, epoch watchdog, compile cache).
- `sandbox`: one running instance of a module: init, step, fuel and wall time per step, trap policy.
- `metrics`: ATE, RPE, rotation error, Sim(3) diagnostic.
- `run`: episodes from disk through a sandbox → report.json.
- `check`: a synthetic-frame smoke test of a module (protocol, fuel).
- `leaderboard`: reports → accuracy-vs-compute table and plot.
"""
