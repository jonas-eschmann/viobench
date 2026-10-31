"""The Python runtime's packaging step: a solution directory -> one self-contained solution.wasm, and back.

`viobench bundle <dir> -o x.wasm` zips the directory and embeds it into runtime.wasm as a data segment (`embed`);
`viobench unbundle` extracts it. Depends only on the contract (viobench.tools.abi) and the workspaces (viobench.solution.workspace); the runner never
imports this.
"""


class BundleError(Exception):
    """Invalid solution directory or runtime binary."""
