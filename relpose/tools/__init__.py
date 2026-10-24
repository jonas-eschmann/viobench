"""Host side of relpose: everything that runs on your machine around a solution .wasm.

- `abi`       the contract between a solution module and the harness (constants, config, manifest, module inspection)
- `geometry`  SE(3) / quaternion helpers shared by runner and datasets
- `datasets`  episodes: the container, the on-disk format, and the producers (`generate`: HyperDrone, EuRoC)
- `runner`    Engine, Sandbox, metrics, evaluate over a Dataset, leaderboard   — the Runner Library
- `server`    submissions over HTTP → the same runner on hidden datasets → leaderboard
- `cli`       `relpose` command line over all of the above (and `bundle`, the Python runtime's packaging step)

Dependencies point downward only: runner and datasets depend on abi/geometry; server depends on runner; nothing on
the host side knows how a .wasm was produced beyond what the ABI says.
"""
