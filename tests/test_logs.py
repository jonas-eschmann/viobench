"""Bounded logs through real WASI writes and bounded reads of existing large logs."""
from pathlib import Path

import numpy as np
import wasmtime

from viobench import Sandbox, abi
from viobench.tools.abi.config import default_guest_config
from viobench.tools.runner.logs import MAX_LOG_BYTES, MAX_TAIL_BYTES
from viobench.tools.runner.sandbox import tail


def test_guest_streams_keep_bounded_prefixes_across_calls(engine, tmp_path):
    chunk = 700 << 10
    wasm = tmp_path / "noisy.wasm"
    wasm.write_bytes(wasmtime.wat2wasm(f"""
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 32)
  (global $next (mut i32) (i32.const 1048576))
  (func (export "abi_version") (result i32) (i32.const {abi.ABI_VERSION}))
  (func (export "alloc") (param $size i32) (result i32) (local $pointer i32)
    (local.set $pointer (global.get $next))
    (global.set $next (i32.and (i32.add (i32.add (global.get $next) (local.get $size)) (i32.const 7)) (i32.const -8)))
    (local.get $pointer))
  (func $write (param $fd i32) (param $byte i32) (param $size i32)
    (memory.fill (i32.const 4096) (local.get $byte) (local.get $size))
    (i32.store (i32.const 0) (i32.const 4096))
    (i32.store (i32.const 4) (local.get $size))
    (if (call $fd_write (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 8)) (then unreachable))
    ;; Discarded bytes must still be acknowledged, so guests do not retry forever.
    (if (i32.ne (i32.load (i32.const 8)) (local.get $size)) (then unreachable)))
  (func (export "_initialize")
    (call $write (i32.const 1) (i32.const 79) (i32.const 7))
    (call $write (i32.const 2) (i32.const 69) (i32.const 11)))
  (func (export "init") (param i32 i32) (result i32)
    (call $write (i32.const 1) (i32.const 73) (i32.const {chunk}))
    (call $write (i32.const 2) (i32.const 105) (i32.const {chunk}))
    (i32.const 0))
  (func (export "step") (param f64 i32 i32 i32) (param $out i32) (result i32)
    (call $write (i32.const 1) (i32.const 83) (i32.const {chunk}))
    (call $write (i32.const 2) (i32.const 115) (i32.const {chunk}))
    (memory.fill (local.get $out) (i32.const 0) (i32.const {abi.STATE_VALUES * 8}))
    (f64.store offset=48 (local.get $out) (f64.const 1))
    (i32.const 0)))
"""))
    expected_stdout = b"O" * 7 + b"I" * chunk + b"S" * (MAX_LOG_BYTES - chunk - 7)
    expected_stderr = b"E" * 11 + b"i" * chunk + b"s" * (MAX_LOG_BYTES - chunk - 11)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(2):  # A new episode gets fresh budgets, including when reusing a log directory.
        sandbox = Sandbox(wasm, engine, log_dir=tmp_path / "logs")
        assert sandbox.stdout_path.read_bytes() == b"O" * 7
        assert sandbox.stderr_path.read_bytes() == b"E" * 11
        sandbox.init(default_guest_config())
        assert sandbox.stdout_path.stat().st_size == chunk + 7
        assert sandbox.stderr_path.stat().st_size == chunk + 11
        for t in (0.05, 0.10):
            sandbox.step(t, frame, np.empty((0, 7)))
            assert sandbox.stdout_path.read_bytes() == expected_stdout
            assert sandbox.stderr_path.read_bytes() == expected_stderr


def test_tail_handles_missing_empty_zero_lines_and_invalid_utf8(tmp_path):
    path = tmp_path / "log.txt"
    assert tail(path) == ""
    path.write_bytes(b"")
    assert tail(path) == ""
    path.write_bytes(b"first\ncaf\xc3\xa9\nbad:\xff\nlast\n")
    assert tail(path, 0) == ""
    assert tail(path, 2) == "bad:\ufffd\nlast"
    assert tail(path) == "first\ncaf\u00e9\nbad:\ufffd\nlast"


def test_tail_reads_only_the_final_byte_window(tmp_path, monkeypatch):
    path = tmp_path / "large.log"
    # A sparse file makes a full-file read dangerous without allocating a giant test input.
    retained = b"\xa9" + b"x" * (MAX_TAIL_BYTES - 6) + b"\nend\n"
    with path.open("wb") as stream:
        stream.seek(1 << 30)
        stream.write(b"\xc3" + retained)  # The byte window cuts a valid UTF-8 character in half.
    size = path.stat().st_size
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, count=-1):
            assert self.stream.tell() >= size - MAX_TAIL_BYTES  # Fail before an oversized read can allocate.
            content = self.stream.read(count)
            reads.append(len(content))
            assert len(content) <= MAX_TAIL_BYTES
            return content

    def guarded_open(opened, *args, **kwargs):
        stream = original_open(opened, *args, **kwargs)
        return BoundedReader(stream) if opened == path else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    assert tail(path) == retained.decode("utf-8", errors="replace").rstrip("\n")
    assert tail(path, 1) == "end"
    assert reads == [MAX_TAIL_BYTES, MAX_TAIL_BYTES]
