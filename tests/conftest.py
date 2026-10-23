import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNTIME = Path(os.environ.get("EECS116_RUNTIME", REPO / "build" / "wasm-python" / "dist" / "runtime.wasm"))


@pytest.fixture(scope="session")
def runtime_path():
    if not RUNTIME.is_file():
        pytest.skip(f"runtime.wasm not built ({RUNTIME}); run build_wasm_python.sh")
    return RUNTIME


@pytest.fixture(scope="session")
def runtime():
    """One compiled engine for the whole session (compiling the 40 MiB module takes seconds)."""
    from eecs116 import Runtime

    return Runtime()


@pytest.fixture
def solution_dir(tmp_path):
    """Factory: write solution.py (and extra files) into a fresh directory and return it."""

    def make(source, extra=None):
        directory = tmp_path / "solution"
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
