"""Packaging: a solution directory → one self-contained solution.wasm (and back).

- `wasm_binary`: generic reader/rewriter for WebAssembly sections (custom, memory, data).
- `embed`: the data-segment mechanism: put a zip above the Python runtime's initial memory, patch its descriptor.
- `manifest`: what a module declares to the harness before it runs (entry point, channels, resolution).
- `bundle`: directory validation + zip + embed + manifest, i.e. `relpose bundle` / `relpose unbundle`.

Native modules need none of this except, optionally, `relpose manifest` (see submission/native/).
"""


class PackagingError(Exception):
    """Invalid solution directory, manifest or module binary."""
