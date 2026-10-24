import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNTIME = Path(os.environ.get("RELPOSE_RUNTIME", REPO / "build" / "runtime" / "dist" / "runtime.wasm"))


@pytest.fixture(scope="session")
def runtime_path():
    if not RUNTIME.is_file():
        pytest.skip(f"runtime.wasm not built ({RUNTIME}); run relpose/solution/runtime/python/build.sh")
    return RUNTIME


@pytest.fixture(scope="session")
def engine():
    """One wasmtime engine (with its compile cache) for the whole session."""
    from relpose import Engine

    return Engine()


@pytest.fixture
def solution_dir(tmp_path):
    """Factory: write solution.py (and extra files) into a fresh directory and return it."""

    def make(source, extra=None, name="solution"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "solution.py").write_text(source)
        for name, content in (extra or {}).items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
        make.directory = directory
        return directory

    return make
