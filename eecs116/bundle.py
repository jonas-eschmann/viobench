"""Turn a directory of Python code (plus data files) into one solution .wasm, and back.

The directory is zipped (uncompressed, reproducible) and embedded into the prebuilt runtime.wasm as a *data segment*
placed just above the runtime's initial memory (the memory minimum grows accordingly); the runtime's bundle descriptor
(runtime/reactor.c, found through its magic bytes) is patched with the zip's address and length. The module is then
self-contained: at init the frozen shim (runtime/py/_eecs116.py) finds the zip in its own memory and imports from it.
The manifest (entry point, channels, resolution) is also stored in a custom section so the harness can read it before
instantiation. Nothing is compiled, so no toolchain is needed.
"""
import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import struct
from dataclasses import dataclass

from . import abi
from .wasm_sections import (DATA_COUNT_SECTION_ID, DATA_SECTION_ID, MEMORY_SECTION_ID, PAGE_BYTES, DataSegment, append_custom,
                            encode_data_section, encode_memory_section, find_custom, parse_data_section, parse_memory_section,
                            read_leb128, replace_sections, section_payload, strip_custom, write_leb128)

MANIFEST_KEYS = {"entry", "channels", "resolution", "format"}


@dataclass
class Descriptor:
    segment: int  # index of the data segment holding the descriptor
    position: int  # byte offset of the descriptor inside that segment's data
    memory_address: int  # linear-memory address of the descriptor
    address: int  # linear-memory address of the embedded zip (0 = none)
    length: int  # bytes of the embedded zip (0 = none)
    version: int


def runtime_descriptor(wasm):
    """The runtime's bundle descriptor, or None for a module without one (a native solution)."""
    payload = section_payload(wasm, DATA_SECTION_ID)
    if payload is None:
        return None
    found = None
    for index, segment in enumerate(parse_data_section(payload)):
        position = segment.data.find(abi.BUNDLE_MAGIC)
        while position != -1:
            if found is not None:
                raise BundleError("the module contains more than one bundle descriptor")
            if segment.offset is None:
                raise BundleError("the bundle descriptor lives in a segment with a non-constant offset")
            address, length, version, _ = struct.unpack_from("<IIII", segment.data, position + len(abi.BUNDLE_MAGIC))
            found = Descriptor(index, position, segment.offset + position, address, length, version)
            position = segment.data.find(abi.BUNDLE_MAGIC, position + 1)
    return found


def _memory(wasm):
    payload = section_payload(wasm, MEMORY_SECTION_ID)
    if payload is None:
        raise BundleError("the module defines no memory (imported memories are not supported)")
    memories = parse_memory_section(payload)
    if len(memories) != 1:
        raise BundleError(f"expected one memory, found {len(memories)}")
    return memories


def _patched_descriptor(segments, descriptor, address, length):
    segment = segments[descriptor.segment]
    data = bytearray(segment.data)
    struct.pack_into("<II", data, descriptor.position + len(abi.BUNDLE_MAGIC), address, length)
    segments[descriptor.segment] = DataSegment(segment.flags, segment.memory, segment.offset, segment.offset_expression, bytes(data))


def _rebuild(wasm, memories, segments):
    payloads = {MEMORY_SECTION_ID: encode_memory_section(memories), DATA_SECTION_ID: encode_data_section(segments)}
    if section_payload(wasm, DATA_COUNT_SECTION_ID) is not None:
        payloads[DATA_COUNT_SECTION_ID] = write_leb128(len(segments))
    return replace_sections(wasm, payloads)


def strip_bundle(wasm):
    """The runtime without an embedded zip (the given binary if it has none)."""
    descriptor = runtime_descriptor(wasm)
    if descriptor is None or descriptor.length == 0:
        return wasm
    memories = _memory(wasm)
    segments = parse_data_section(section_payload(wasm, DATA_SECTION_ID))
    kept = [s for s in segments if not (s.offset == descriptor.address and len(s.data) == descriptor.length)]
    if len(kept) != len(segments) - 1:
        raise BundleError("the embedded bundle segment does not match its descriptor")
    memories[0].minimum = descriptor.address // PAGE_BYTES
    _patched_descriptor(kept, descriptor, 0, 0)
    return _rebuild(wasm, memories, kept)


def extract_bundle(wasm):
    """The embedded zip bytes, or None."""
    descriptor = runtime_descriptor(wasm)
    if descriptor is None or descriptor.length == 0:
        return None
    for segment in parse_data_section(section_payload(wasm, DATA_SECTION_ID)):
        if segment.offset == descriptor.address and len(segment.data) == descriptor.length:
            return segment.data
    raise BundleError("the embedded bundle segment does not match its descriptor")


def embed_bundle(wasm, payload):
    """The runtime with payload embedded above its initial memory and the descriptor pointing at it."""
    wasm = strip_bundle(wasm)
    descriptor = runtime_descriptor(wasm)
    if descriptor is None:
        raise BundleError("not a Python runtime: the module has no bundle descriptor (bundle on runtime.wasm)")
    memories = _memory(wasm)
    address = memories[0].minimum * PAGE_BYTES  # just above the heap's initial arena; growth continues beyond the zip
    pages = (len(payload) + PAGE_BYTES - 1) // PAGE_BYTES
    memories[0].minimum += pages
    if memories[0].maximum is not None and memories[0].maximum < memories[0].minimum:
        memories[0].maximum = memories[0].minimum
    segments = parse_data_section(section_payload(wasm, DATA_SECTION_ID))
    segments.append(DataSegment.active(address, payload))
    _patched_descriptor(segments, descriptor, address, len(payload))
    return _rebuild(wasm, memories, segments)

IGNORED_DIRECTORIES = {"__pycache__", ".git", ".hg", ".svn", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "node_modules"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
SIZE_WARNING_BYTES = 64 << 20
RUNTIME_VARIABLE = "EECS116_RUNTIME"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class BundleError(Exception):
    pass


def default_runtime():
    """runtime.wasm from $EECS116_RUNTIME, next to this package, or the repository build directory."""
    candidates = []
    if os.environ.get(RUNTIME_VARIABLE):
        candidates.append(Path(os.environ[RUNTIME_VARIABLE]))
    here = Path(__file__).resolve().parent
    candidates += [here / "runtime.wasm", here.parent / "build" / "wasm-python" / "dist" / "runtime.wasm"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise BundleError(f"runtime.wasm not found; pass --runtime or set {RUNTIME_VARIABLE} (looked at {[str(c) for c in candidates]})")


def collect_files(directory):
    """Relative paths of everything that goes into the bundle, sorted; refuses symlinks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise BundleError(f"{directory} is not a directory")
    files = []
    for root, directories, names in os.walk(directory):
        directories[:] = sorted(d for d in directories if d not in IGNORED_DIRECTORIES and not d.startswith("."))
        for name in sorted(names):
            path = Path(root) / name
            if name.startswith(".") or path.suffix in IGNORED_SUFFIXES:
                continue
            if path.is_symlink():
                raise BundleError(f"{path} is a symbolic link; copy the file into the directory instead")
            files.append(path.relative_to(directory))
    if not files:
        raise BundleError(f"{directory} is empty")
    return files


def entry_file(files, module_name):
    relative = module_name.replace(".", "/")
    names = {file.as_posix() for file in files}
    for candidate in (f"{relative}/__init__.py", f"{relative}.py"):
        if candidate in names:
            return candidate
    return None


def validate(directory, files, entry):
    """Syntax-checks every .py and looks for the entry class; returns warnings (hard errors raise)."""
    directory = Path(directory)
    module_name, _, class_name = entry.partition(":")
    class_name = class_name or "Solution"
    warnings = []
    errors = []
    trees = {}
    for file in files:
        if file.suffix != ".py":
            continue
        source = (directory / file).read_bytes()
        try:
            trees[file.as_posix()] = ast.parse(source, filename=str(file))
        except SyntaxError as error:
            errors.append(f"{file}:{error.lineno}: {error.msg}")
    if errors:
        raise BundleError("syntax errors:\n  " + "\n  ".join(errors))
    main = entry_file(files, module_name)
    if main is None:
        raise BundleError(f"entry module {module_name!r} not found: expected {module_name.replace('.', '/')}.py or {module_name.replace('.', '/')}/__init__.py")
    classes = {node.name: node for node in ast.walk(trees[main]) if isinstance(node, ast.ClassDef)}
    if class_name not in classes:
        warnings.append(f"{main} does not define class {class_name} itself (fine if it imports it from another module)")
    else:
        methods = {node.name for node in classes[class_name].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for method in ("__init__", "step"):
            if method not in methods:
                warnings.append(f"class {class_name} in {main} has no {method} method (fine if inherited)")
    total = sum((directory / file).stat().st_size for file in files)
    if total > SIZE_WARNING_BYTES:
        warnings.append(f"bundle is {total / 2**20:.0f} MiB; the submission limit may be lower")
    return warnings


def read_manifest(wasm):
    """Manifest dict of a solution binary: the eecs116.manifest section, else eecs116.json inside the bundle, else {}."""
    section = find_custom(wasm, abi.MANIFEST_SECTION)
    if section is not None:
        return json.loads(section.decode("utf-8"))
    payload = extract_bundle(wasm)
    if payload is not None:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if abi.BUNDLE_MANIFEST in archive.namelist():
                return json.loads(archive.read(abi.BUNDLE_MANIFEST).decode("utf-8"))
    return {}


def parse_resolution(value):
    """'640x480' | [640, 480] | (640, 480) -> (640, 480)"""
    if isinstance(value, str):
        parts = value.lower().replace("×", "x").split("x")
        if len(parts) != 2:
            raise BundleError(f"resolution must look like 640x480, got {value!r}")
        value = parts
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError):
        raise BundleError(f"resolution must be [width, height], got {value!r}") from None
    return width, height


def validate_manifest(manifest):
    channels = manifest.get("channels", abi.DEFAULT_CHANNELS)
    if channels not in abi.SUPPORTED_CHANNELS:
        raise BundleError(f"manifest channels must be one of {abi.SUPPORTED_CHANNELS}, got {channels!r}")
    if "resolution" in manifest:
        resolution = parse_resolution(manifest["resolution"])
        if resolution not in abi.SUPPORTED_RESOLUTIONS:
            raise BundleError(f"manifest resolution must be one of {['x'.join(map(str, r)) for r in abi.SUPPORTED_RESOLUTIONS]}, got {manifest['resolution']!r}")
        manifest["resolution"] = list(resolution)
    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        raise BundleError(f"unknown manifest keys {sorted(unknown)} (known: {sorted(MANIFEST_KEYS)})")
    return manifest


def write_manifest(wasm_path, updates):
    """Merges updates into the binary's manifest section (for native modules: `eecs116 manifest x.wasm --channels 3`)."""
    wasm_path = Path(wasm_path)
    wasm = wasm_path.read_bytes()
    manifest = validate_manifest({**read_manifest(wasm), **updates})
    wasm_path.write_bytes(append_custom(wasm, abi.MANIFEST_SECTION, json.dumps(manifest, sort_keys=True).encode("utf-8")))
    return manifest


def make_zip(directory, files, entry, channels=abi.DEFAULT_CHANNELS, resolution=abi.DEFAULT_RESOLUTION):
    """Uncompressed (the guest has no zlib), deterministic zip with the manifest at the end."""
    directory = Path(directory)
    manifest = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            name = file.as_posix()
            data = (directory / file).read_bytes()
            if name == abi.BUNDLE_MANIFEST:
                manifest = json.loads(data)
                continue
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
        manifest.update({"format": 1, "entry": entry, "channels": int(channels), "resolution": list(parse_resolution(resolution))})
        validate_manifest(manifest)
        info = zipfile.ZipInfo(abi.BUNDLE_MANIFEST, date_time=ZIP_EPOCH)
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(manifest, indent=1, sort_keys=True))
    return buffer.getvalue(), manifest


def bundle(directory, output, runtime=None, entry=None, channels=None, resolution=None):
    """Writes output (.wasm) and returns a summary dict. entry/channels/resolution default to eecs116.json in the directory."""
    directory = Path(directory)
    files = collect_files(directory)
    manifest_path = directory / abi.BUNDLE_MANIFEST
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if entry is None:
        entry = existing.get("entry", abi.DEFAULT_ENTRY)
    if channels is None:
        channels = existing.get("channels", abi.DEFAULT_CHANNELS)
    if resolution is None:
        resolution = existing.get("resolution", abi.DEFAULT_RESOLUTION)
    warnings = validate(directory, files, entry)
    runtime_path = Path(runtime) if runtime else default_runtime()
    runtime_bytes = strip_custom(runtime_path.read_bytes(), abi.MANIFEST_SECTION)
    payload, manifest = make_zip(directory, files, entry, channels, resolution)
    result = embed_bundle(runtime_bytes, payload)
    result = append_custom(result, abi.MANIFEST_SECTION, json.dumps(manifest, sort_keys=True).encode("utf-8"))
    if extract_bundle(result) != payload or read_manifest(result) != manifest:
        raise BundleError("internal error: the embedded bundle does not round-trip")
    output = Path(output)
    output.write_bytes(result)
    return {
        "output": str(output),
        "bytes": len(result),
        "files": [file.as_posix() for file in files],
        "bundle_bytes": len(payload),
        "entry": entry,
        "channels": int(channels),
        "resolution": manifest["resolution"],
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(strip_bundle(runtime_bytes)).hexdigest(),
        "warnings": warnings,
    }


def unbundle(wasm_path, output_directory):
    """Extracts the bundled directory (for inspection); returns the list of files."""
    payload = extract_bundle(Path(wasm_path).read_bytes())
    if payload is None:
        raise BundleError(f"{wasm_path} carries no embedded bundle")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        for name in names:
            target = (output_directory / name).resolve()
            if output_directory.resolve() not in target.parents:
                raise BundleError(f"refusing to extract {name!r} outside {output_directory}")
        archive.extractall(output_directory)
    return names
