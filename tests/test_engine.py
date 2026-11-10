"""Compilation coordination across independent evaluator processes."""
import multiprocessing
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="worker compilation uses flock")


def _gated_module(wasm, connection):
    from viobench.tools.runner import engine

    def compile_module(*_):
        connection.send("compiling")
        assert connection.poll(15), "parent did not release compilation"
        if connection.recv() == "raise":
            raise ValueError("invalid module")
        return object()

    # Exercise Engine.module and its real file lock without an expensive compiler workload.
    engine.wasmtime.Module = compile_module
    evaluator = engine.Engine.__new__(engine.Engine)
    evaluator.wasmtime, evaluator._modules = None, {}
    connection.send("ready")
    try:
        evaluator.module(wasm)
    except ValueError:
        connection.send("failed")
    else:
        connection.send("loaded")
    assert connection.poll(15), "parent did not finish the evaluator"
    assert connection.recv() == "exit"


def _load_module(wasm, connection):
    from viobench.tools.runner.engine import Engine

    evaluator = Engine()
    module = evaluator.module(wasm)
    assert evaluator.module(wasm) is module
    connection.send([export.name for export in module.exports])


@pytest.fixture
def processes(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOBENCH_COMPILE_LOCK", str(tmp_path / "compile.lock"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    context, children = multiprocessing.get_context("spawn"), []

    def start(wasm, target=_gated_module):
        parent, child = context.Pipe()
        process = context.Process(target=target, args=(wasm, child))
        process.start()
        child.close()
        children.append((process, parent))
        return process, parent

    yield start
    for process, connection in children:
        if process.is_alive():
            process.kill()
        process.join(5)
        connection.close()


def message(connection):
    assert connection.poll(10), "evaluator did not make progress"
    return connection.recv()


@pytest.mark.parametrize("second_wasm", [b"first module", b"different module"])
@pytest.mark.parametrize("finish", ["return", "raise", "kill"])
def test_compilation_is_global_and_released(processes, second_wasm, finish):
    first, first_pipe = processes(b"first module")
    assert message(first_pipe) == "ready"
    assert message(first_pipe) == "compiling"
    second, second_pipe = processes(second_wasm)
    assert message(second_pipe) == "ready"
    assert not second_pipe.poll(0.2), "compilations overlapped"

    if finish == "kill":
        first.kill()
        first.join(5)
        assert first.exitcode is not None
    else:
        first_pipe.send(finish)
        assert message(first_pipe) == ("loaded" if finish == "return" else "failed")
        assert first.is_alive()  # The lock must release before the evaluator exits.

    assert message(second_pipe) == "compiling"
    second_pipe.send("return")
    assert message(second_pipe) == "loaded"
    second_pipe.send("exit")
    second.join(5)
    assert second.exitcode == 0
    if finish != "kill":
        first_pipe.send("exit")
        first.join(5)
        assert first.exitcode == 0


def test_separate_evaluators_reuse_wasmtime_cache(processes, tmp_path):
    import wasmtime

    wasm = bytes(wasmtime.wat2wasm('(module (func (export "answer") (result i32) i32.const 42))'))
    first, pipe = processes(wasm, _load_module)
    assert message(pipe) == ["answer"]
    first.join(5)
    assert first.exitcode == 0
    cache = tmp_path / "cache" / "wasmtime" / "modules"
    artifacts = {path: path.stat().st_ino for path in cache.rglob("*") if path.is_file() and path.suffix != ".stats"}
    assert artifacts, "compilation did not populate the disk cache"

    second, pipe = processes(wasm, _load_module)
    assert message(pipe) == ["answer"]
    second.join(5)
    assert second.exitcode == 0
    # A miss writes a replacement cache file atomically; a hit keeps the compiled artifact.
    assert {path: path.stat().st_ino for path in artifacts} == artifacts
