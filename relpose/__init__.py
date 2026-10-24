"""relpose: compute-aware relative pose estimation challenge.

Two halves, split at the WebAssembly boundary:
- `relpose.tools`    host side — runs, scores, serves: `abi` (the contract), `datasets`, `runner`, `server`, `cli`.
- `relpose.solution` guest side — how a solution .wasm is built: toolchain, runtime/{native,python}, examples.
                     Mostly files, not modules; the Python runtime's bundler is the one importable piece.

`import relpose` gives the host-side essentials: Engine, Sandbox, Limits, StepResult, Episode, read_episode, evaluate.
"""
from .tools import abi
from .tools.runner.engine import Engine
from .tools.runner.sandbox import Limits, Sandbox, SandboxError, SandboxTrap, StepResult

__all__ = ["abi", "Engine", "Limits", "Sandbox", "SandboxError", "SandboxTrap", "StepResult", "Episode", "read_episode", "evaluate"]


def __getattr__(name):  # heavier optional dependencies (cv2, yaml) load on first use
    if name in ("Episode", "read_episode"):
        from .tools import datasets

        return getattr(datasets, name)
    if name == "evaluate":
        from .tools.runner.run import evaluate

        return evaluate
    raise AttributeError(name)
