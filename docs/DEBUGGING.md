# Debugging

No debugger attaches inside wasmtime. Both flavours therefore also run on the host, with the inputs the sandbox would
feed: no fuel and no limits, and your host's libraries. `check` and `evaluate` remain the measure.

## Python

```bash
relpose debug my_solution data/val --max-episodes 1
relpose debug my_solution --simulate 2             # fresh simulated episodes (GPU)
relpose debug my_solution data/val -o out          # writes estimate.tum, twist.txt and trajectory.mp4 under out/<dataset>/<episode>/
```

`breakpoint()` works anywhere in your code; in VS Code, use the launch configuration *relpose: debug Python solution*.
For plots, `trace` yields one record per frame (`t`, `frame`, `imu`, `pose`, `twist`, host seconds):

```python
import numpy as np, matplotlib.pyplot as plt
from relpose.tools.datasets import read_episode
from relpose.tools.runner.direct import PythonSolution, trace

episode = read_episode("data/val/<name>")
poses = np.array([s.pose for s in trace(PythonSolution("my_solution"), episode)])
plt.plot(poses[:, 0], poses[:, 1], label="estimate")
plt.plot(*episode.ground_truth_relative()[:, :2].T, label="truth"); plt.legend(); plt.show()
```

## Native

Configured with the host compiler, the `relpose_solution(...)` CMake call builds `build-host/debug`. It replays a
*feed* (one episode as raw arrays) through `init` and `step`:

```bash
relpose feed data/val/<name> -o build/feed                     # --resolution 640x480 to match frame_width()
cmake -S my_solution -B build-host && cmake --build build-host
build-host/debug build/feed estimate.tum                       # stops at the first failing step
relpose score data/val/<name> estimate.tum                     # reads twist.txt next to it
gdb --args build-host/debug build/feed                         # VS Code: "relpose: debug native solution (gdb)"
```

`relpose feed --simulate -o build/feed` feeds a fresh simulated episode.

## Host vs. sandbox

| | sandbox (`check`, `evaluate`) | host (`debug`) |
|---|---|---|
| compute | fuel, limits, memory cap | none |
| libraries | the runtime's, or what you linked | what is installed on your machine |
| failures | end the episode | raise in your process |
| inputs | identical | identical |

When something works on the host but not in `check`, `relpose check -v` shows the guest's stderr.
