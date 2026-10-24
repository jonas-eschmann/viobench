"""The Python runtime's packaging step: a solution directory -> one self-contained solution.wasm, and back.

`relpose bundle <dir> -o x.wasm` zips the directory and embeds it into runtime.wasm as a data segment (`embed`);
`relpose unbundle` extracts it. Depends only on the contract (relpose.tools.abi); the runner never imports this.
"""


class BundleError(Exception):
    """Invalid solution directory or runtime binary."""
