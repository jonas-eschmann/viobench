"""Turn a directory of Python code (plus data files) into one solution .wasm, and back.

The directory is zipped (uncompressed, reproducible) and appended to the prebuilt runtime.wasm as a custom section
named eecs116.bundle. Nothing in the module's code changes, so no toolchain is needed. The guest-side shim
(runtime/py/_eecs116.py) imports straight from the zip.
"""
import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from . import abi
from .wasm_sections import append_custom, find_custom, strip_custom

MANIFEST_KEYS = {"entry", "channels", "format"}

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
    payload = find_custom(wasm, abi.BUNDLE_SECTION)
    if payload is not None:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if abi.BUNDLE_MANIFEST in archive.namelist():
                return json.loads(archive.read(abi.BUNDLE_MANIFEST).decode("utf-8"))
    return {}


def validate_manifest(manifest):
    channels = manifest.get("channels", abi.DEFAULT_CHANNELS)
    if channels not in abi.SUPPORTED_CHANNELS:
        raise BundleError(f"manifest channels must be one of {abi.SUPPORTED_CHANNELS}, got {channels!r}")
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


def make_zip(directory, files, entry, channels=abi.DEFAULT_CHANNELS):
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
        manifest.update({"format": 1, "entry": entry, "channels": int(channels)})
        validate_manifest(manifest)
        info = zipfile.ZipInfo(abi.BUNDLE_MANIFEST, date_time=ZIP_EPOCH)
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(manifest, indent=1, sort_keys=True))
    return buffer.getvalue(), manifest


def bundle(directory, output, runtime=None, entry=None, channels=None):
    """Writes output (.wasm) and returns a summary dict. entry/channels default to eecs116.json in the directory."""
    directory = Path(directory)
    files = collect_files(directory)
    manifest_path = directory / abi.BUNDLE_MANIFEST
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if entry is None:
        entry = existing.get("entry", abi.DEFAULT_ENTRY)
    if channels is None:
        channels = existing.get("channels", abi.DEFAULT_CHANNELS)
    warnings = validate(directory, files, entry)
    runtime_path = Path(runtime) if runtime else default_runtime()
    runtime_bytes = runtime_path.read_bytes()
    base = strip_custom(strip_custom(runtime_bytes, abi.BUNDLE_SECTION), abi.MANIFEST_SECTION)
    payload, manifest = make_zip(directory, files, entry, channels)
    result = append_custom(base, abi.BUNDLE_SECTION, payload)
    result = append_custom(result, abi.MANIFEST_SECTION, json.dumps(manifest, sort_keys=True).encode("utf-8"))
    if find_custom(result, abi.BUNDLE_SECTION) != payload or read_manifest(result) != manifest:
        raise BundleError("internal error: bundle sections do not round-trip")
    output = Path(output)
    output.write_bytes(result)
    return {
        "output": str(output),
        "bytes": len(result),
        "files": [file.as_posix() for file in files],
        "bundle_bytes": len(payload),
        "entry": entry,
        "channels": int(channels),
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(base).hexdigest(),
        "warnings": warnings,
    }


def unbundle(wasm_path, output_directory):
    """Extracts the bundled directory (for inspection); returns the list of files."""
    payload = find_custom(Path(wasm_path).read_bytes(), abi.BUNDLE_SECTION)
    if payload is None:
        raise BundleError(f"{wasm_path} carries no {abi.BUNDLE_SECTION} section")
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
