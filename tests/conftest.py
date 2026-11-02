import hashlib
import os
import shutil
from pathlib import Path

import pytest

from viobench.solution import workspace

RUNTIME = Path(os.environ.get("VIOBENCH_RUNTIME") or workspace.runtime_path(workspace.PACKAGE))


@pytest.fixture(scope="session")
def runtime_path():
    if not RUNTIME.is_file():
        pytest.skip(f"runtime.wasm not built ({RUNTIME}); run viobench build-runtime")
    return RUNTIME


@pytest.fixture(scope="session")
def engine():
    """One wasmtime engine (with its compile cache) for the whole session."""
    from viobench import Engine

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


@pytest.fixture
def conta_store(tmp_path, monkeypatch):
    """An offline conta store for one test (CONTA_ROOT: no network, nothing in ~/.cache/conta).
    Returns add(path), which stores the file under its sha1 and returns the sha1."""
    root = tmp_path / "conta-store"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("CONTA_ROOT", str(root))
    monkeypatch.setenv("CONTA_CACHE", str(tmp_path / "conta-cache"))
    monkeypatch.delenv("CONTA_URL", raising=False)

    def add(path):
        digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        shutil.copyfile(path, root / "data" / digest)
        return digest

    return add
