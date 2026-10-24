"""relpose: host-side toolkit of the relative pose-estimation challenge.

Subpackages mirror the architecture: `abi` (the contract), `packaging` (directory -> solution.wasm), `evaluation`
(solution.wasm -> numbers), `datasets` (episodes: storage, generation, real data), `server` (submissions).
"""
from . import abi
from .evaluation.engine import Engine
from .evaluation.sandbox import Limits, Sandbox, SandboxError, SandboxTrap, StepResult

__all__ = ["abi", "Engine", "Limits", "Sandbox", "SandboxError", "SandboxTrap", "StepResult", "Episode", "read_episode", "evaluate"]


def __getattr__(name):  # heavier optional dependencies (cv2, yaml) load on first use
    if name in ("Episode", "read_episode"):
        from . import datasets

        return getattr(datasets, name)
    if name == "evaluate":
        from .evaluation.run import evaluate

        return evaluate
    raise AttributeError(name)
