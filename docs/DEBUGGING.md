# Debugging a solution outside the sandbox

Inside wasmtime nothing can attach a debugger, and `relpose check` only shows you captured stderr. Both solution
flavours therefore have a second way to run: **the same code, driven by the same inputs the sandbox would feed, in
an ordinary host process** where `breakpoint()`, gdb, prints and matplotlib work. Nothing is emulated: no fuel, no
limits, your host's numpy/OpenCV instead of the runtime's. `relpose check` and `relpose evaluate` stay the authority on
what the sandbox accepts and what it costs.

## Python: `relpose debug` / `python solution.py`

```bash
relpose debug my_solution build/datasets/public/default-val --episode-limit 1       # the directory with solution.py / solution.json
python my_solution/solution.py build/datasets/public/default-val --episode-limit 1  # the same, through the examples' footer
relpose debug my_solution --simulate 2 --condition blur                             # fresh HyperDrone episodes (GPU)
relpose debug my_solution <episode dir> -o out       # writes out/<dataset>/<episode>/estimate.tum like `evaluate` does
```

The driver imports your directory (`solution.json` is honoured: `entry`, `resolution`), provides the
`solution_bundle` module backed by the files next to your code, adapts each episode to your resolution (RGB frames) and
calls `Solution(config)` then `step(t, frame, imu)` per frame, exactly like the runtime. It prints the metrics line of
`evaluate` (without fuel) per episode.

- **Breakpoints**: put `breakpoint()` anywhere in `step`; pdb opens in the terminal. In VS Code pick the
  *relpose: debug Python solution* launch configuration (`.vscode/launch.json`, edit the paths), set breakpoints, F5.
- **Prints** go to your terminal, uncaptured.
- **Plots and custom loops**: `trace` yields one record per frame (`t`, `frame`, `imu`, `pose`, host seconds):

  ```python
  import numpy as np, matplotlib.pyplot as plt
  from relpose.tools.datasets import read_episode
  from relpose.tools.runner.direct import PythonSolution, trace

  episode = read_episode("build/datasets/public/default-val/<name>")
  poses = np.array([s.pose for s in trace(PythonSolution("my_solution"), episode) if s.pose is not None])
  plt.plot(poses[:, 0], poses[:, 1], label="estimate"); gt = episode.ground_truth_relative(); plt.plot(gt[:, 0], gt[:, 1], label="truth"); plt.legend(); plt.show()
  ```

The footer that makes `python solution.py …` work is three lines at the end of every example and costs nothing inside
the wasm (the runtime imports the module, it never runs it as `__main__`). One caveat for that form: the module body
runs before the footer registers `solution_bundle`, so a solution with data files imports it inside `__init__` (as
`tiny_cnn` does); `relpose debug <dir>` has no such constraint.

```python
if __name__ == "__main__":
    from relpose.tools.runner.direct import main
    main(Solution)
```

## Native: the `debug` CMake target

`relpose_solution(<target> <sources…>)` in your `CMakeLists.txt` builds the submission when configured with the
wasi-sdk toolchain (`runtime/native/build.sh`) and, when configured with your host compiler, an executable `debug`:
your sources plus `relpose_debug_main.c`, compiled with `-g -O1`. That driver reads a *feed* (one episode's inputs as
raw arrays, so C needs no PNG or JSON parser), calls `init` and `step` like the sandbox (same IMU windows
`t_prev < t ≤ t_frame`, same buffers) and writes `estimate.tum`.

```bash
relpose feed build/datasets/public/default-val/<name> -o build/feed          # --resolution 640x480 if your solution declares it
cd my_native_solution && cmake -S . -B build-host && cmake --build build-host
build-host/debug ../build/feed estimate.tum                 # prints per-frame failures, ms/frame and the position RMSE
relpose score build/datasets/public/default-val/<name> estimate.tum          # the full metrics line
gdb --args build-host/debug ../build/feed                   # or lldb; VS Code: "relpose: debug native solution (gdb)"
```

`relpose feed --simulate -o build/feed --condition lowlight` feeds a fresh HyperDrone episode instead. A feed is
`config.json` (the bytes `init` receives), `frames.u8` (N×H×W×3 RGB), `frame_times.f64`, `imu.f64` (M×7, the rows the
harness delivers), `gt.f64` (N×7 relative ground truth) and `shape.txt` (`N H W C M`, C = 3); about 280 MB for a
one-minute episode at 320×240.

With the VS Code CMake Tools extension pointed at your solution directory (`.vscode/settings.json`), the host
configure is the default one, so its *Debug* button launches `debug` under gdb; pass the feed directory as the
program argument in `.vscode/launch.json` or `settings.json` (`cmake.debugConfig.args`).

## What differs from the sandbox

| | sandbox (`check`, `evaluate`) | host (`debug`, `build-host/debug`) |
|---|---|---|
| compute | fuel per frame, step/init limits, memory cap | host time only; no limits |
| libraries | the runtime's numpy/OpenCV (`docs/RUNTIME.md`), or what you linked into your .wasm | whatever is installed on your machine |
| failures | traps end the episode; exceptions become status −1 | exceptions propagate to you (that is the point) |
| inputs | identical: `Episode.adapt` → `config`, frames, IMU windows | identical |

A solution that works on the host but not in `check` is usually using a module the runtime lacks, more memory than
the cap, or a native library; the sandbox's stderr (`relpose check -v`) says which.
