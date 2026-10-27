"""The manifest: what a module declares to the harness before it runs.

`{"entry": "module:Class", "resolution": [320, 240] | [640, 480]}` — stored as the custom section
`relpose.manifest`. Reading is part of the contract (the runner needs it before instantiation); writing is done by the
Python runtime's bundler or by `relpose manifest` for native modules. Python solutions write it with `solution.json` next to `solution.py` (the bundler copies it into
the section); native modules add it with `relpose manifest solution.wasm --resolution 640x480`. Missing keys mean the defaults
(320×240, `solution:Solution`). Frames are always RGB; there is nothing to choose about colour.
"""
import io
import json
import zipfile
from pathlib import Path

from . import BUNDLE_MANIFEST, MANIFEST_SECTION, SUPPORTED_RESOLUTIONS, AbiError
from .wasm_binary import append_custom, find_custom

MANIFEST_KEYS = {"entry", "resolution"}


def parse_resolution(value):
    """'640x480' | [640, 480] | (640, 480) -> (640, 480)"""
    if isinstance(value, str):
        parts = value.lower().replace("×", "x").split("x")
        if len(parts) != 2:
            raise AbiError(f"resolution must look like 640x480, got {value!r}")
        value = parts
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError):
        raise AbiError(f"resolution must be [width, height], got {value!r}") from None
    return width, height


def validate_manifest(manifest):
    if "resolution" in manifest:
        resolution = parse_resolution(manifest["resolution"])
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise AbiError(f"manifest resolution must be one of {['x'.join(map(str, r)) for r in SUPPORTED_RESOLUTIONS]}, got {manifest['resolution']!r}")
        manifest["resolution"] = list(resolution)
    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        raise AbiError(f"unknown manifest keys {sorted(unknown)} (known: {sorted(MANIFEST_KEYS)})")
    return manifest


def read_manifest(wasm):
    """Manifest dict of a module: the relpose.manifest section, else solution.json inside an embedded bundle, else {}."""
    section = find_custom(wasm, MANIFEST_SECTION)
    if section is not None:
        return json.loads(section.decode("utf-8"))
    from .module import embedded_bundle

    payload = embedded_bundle(wasm)
    if payload is not None:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if BUNDLE_MANIFEST in archive.namelist():
                return json.loads(archive.read(BUNDLE_MANIFEST).decode("utf-8"))
    return {}


def with_manifest(wasm, manifest):
    """The binary with its manifest section replaced."""
    return append_custom(wasm, MANIFEST_SECTION, json.dumps(validate_manifest(dict(manifest)), sort_keys=True).encode("utf-8"))


def write_manifest(wasm_path, updates):
    """Merges updates into the module's manifest section (how native modules declare RGB / VGA)."""
    wasm_path = Path(wasm_path)
    wasm = wasm_path.read_bytes()
    manifest = validate_manifest({**read_manifest(wasm), **updates})
    wasm_path.write_bytes(with_manifest(wasm, manifest))
    return manifest
