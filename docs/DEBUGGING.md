# Debugging

No debugger attaches inside wasmtime. Both flavours therefore also run on the host, with the inputs the sandbox would
feed: no fuel and no limits, and your host's libraries. `check` and `evaluate` remain the measure.

## Python

```bash
viobench debug solution --max-episodes 1           # the public set, like evaluate; or name episodes, directories, conta hashes
viobench debug solution -o out                     # writes estimate.tum, twist.txt and trajectory.mp4 under out/<dataset>/<episode>/
```

`breakpoint()` works anywhere in your code; in VS Code, use the launch configuration *viobench: debug Python solution*.
For plots, `trace` yields one record per frame (`t`, `frame`, `imu`, `pose`, `twist`, host seconds):

```python
import numpy as np, matplotlib.pyplot as plt
from viobench.episodes import public
from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.direct import PythonSolution, trace

episode = next(episodes(public[0]))                # or a path: episodes("data/train/<name>.zip")
poses = np.array([s.pose for s in trace(PythonSolution("solution"), episode)])
plt.plot(poses[:, 0], poses[:, 1], label="estimate")
plt.plot(*episode.ground_truth_relative()[:, :2].T, label="truth"); plt.legend(); plt.show()
```

## Native

Configured with the host compiler, the `viobench_solution(...)` CMake call builds `build-host/debug`. It replays a
*feed* (one episode as raw arrays) through `init` and `step`:

```bash
viobench feed "$(conta <sha1>)" -o build/feed                       # a public episode (or a path); --resolution 640x480 to match frame_width()
cmake -S solution -B build-host && cmake --build build-host
build-host/debug build/feed estimate.tum                       # stops at the first failing step
viobench score "$(conta <sha1>)" estimate.tum                       # the same episode; reads twist.txt next to it
gdb --args build-host/debug build/feed                         # VS Code: "viobench: debug native solution (gdb)"
```

Any episode directory works, including a freshly simulated one (`viobench simulate -o build/sim --episodes 1 --no-video`).

## Host vs. sandbox

| | sandbox (`check`, `evaluate`) | host (`debug`) |
|---|---|---|
| compute | fuel, limits, memory cap | none |
| libraries | the runtime's, or what you linked | what is installed on your machine |
| failures | end the episode | raise in your process |
| inputs | identical | identical |

When something works on the host but not in `check`, `viobench check -v` shows the guest's stderr.
