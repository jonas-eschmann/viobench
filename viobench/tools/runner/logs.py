"""Bounded diagnostic logs: retain a prefix when writing and read only a small suffix for reports."""
from pathlib import Path

MAX_LOG_BYTES = 1 << 20  # per stdout/stderr file or worker.txt
MAX_TAIL_BYTES = 16 << 10


def capped_output(path):
    """A WASI output callback: save at most 1 MiB, discard the rest and acknowledge every byte.

    The callback owns the unbuffered file, so tails see writes immediately and Wasmtime releases the file with it.
    """
    output = Path(path).open("wb", buffering=0)
    remaining = MAX_LOG_BYTES

    def write(data):
        nonlocal remaining
        if remaining:
            remaining -= output.write(memoryview(data)[:remaining])
        return len(data)  # discarding output must not make the guest retry it

    return write


def tail(path, lines=40):
    """Last lines within at most 16 KiB of the file's end; empty for missing files or nonpositive line counts."""
    if lines <= 0:
        return ""
    try:
        with Path(path).open("rb") as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell() - MAX_TAIL_BYTES))
            text = source.read(MAX_TAIL_BYTES).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
