"""`relpose bundle`: a directory of Python code (plus data files) → one self-contained solution.wasm; `unbundle` reverses it.

The directory is zipped (uncompressed, reproducible) and embedded into the Python runtime as a data segment
(`embed.py`); the manifest (`solution.json` or command-line flags) is written both into the zip and into the
`relpose.manifest` custom section. Nothing is compiled, so no toolchain is needed beyond the runtime itself.
"""
import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from relpose.tools import abi
from relpose.tools.abi.manifest import parse_resolution, read_manifest, validate_manifest, with_manifest
from relpose.tools.abi.wasm_binary import strip_custom

from . import BundleError as PackagingError
from .embed import embed_bundle, extract_bundle, strip_bundle

IGNORED_DIRECTORIES = {"__pycache__", ".git", ".hg", ".svn", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "node_modules"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
SIZE_WARNING_BYTES = 64 << 20
RUNTIME_VARIABLE = "RELPOSE_RUNTIME"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)



def default_runtime():
    """runtime.wasm from $RELPOSE_RUNTIME, next to this package, or the repository build directory."""
    candidates = []
    if os.environ.get(RUNTIME_VARIABLE):
        candidates.append(Path(os.environ[RUNTIME_VARIABLE]))
    repo = Path(__file__).resolve().parents[5]  # bundler/ python/ runtime/ solution/ relpose/ <repo>
    candidates += [repo / "build" / "runtime" / "dist" / "runtime.wasm"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PackagingError(f"runtime.wasm not found; build it with relpose/solution/runtime/python/build.sh, pass --runtime or set {RUNTIME_VARIABLE} "
                         f"(looked at {[str(c) for c in candidates]})")


def collect_files(directory):
    """Relative paths of everything that goes into the bundle, sorted; refuses symlinks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PackagingError(f"{directory} is not a directory")
    files = []
    for root, directories, names in os.walk(directory):
        directories[:] = sorted(d for d in directories if d not in IGNORED_DIRECTORIES and not d.startswith("."))
        for name in sorted(names):
            path = Path(root) / name
            if name.startswith(".") or path.suffix in IGNORED_SUFFIXES:
                continue
            if path.is_symlink():
                raise PackagingError(f"{path} is a symbolic link; copy the file into the directory instead")
            files.append(path.relative_to(directory))
    if not files:
        raise PackagingError(f"{directory} is empty")
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
        raise PackagingError("syntax errors:\n  " + "\n  ".join(errors))
    main = entry_file(files, module_name)
    if main is None:
        raise PackagingError(f"entry module {module_name!r} not found: expected {module_name.replace('.', '/')}.py or {module_name.replace('.', '/')}/__init__.py")
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


def make_zip(directory, files, entry, channels=abi.DEFAULT_CHANNELS, resolution=abi.DEFAULT_RESOLUTION):
    """Uncompressed (the guest has no zlib), deterministic zip with the manifest at the end. Returns (bytes, manifest)."""
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
    """Writes output (.wasm) and returns a summary dict. entry/channels/resolution default to solution.json in the directory."""
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
    result = with_manifest(embed_bundle(runtime_bytes, payload), manifest)
    if extract_bundle(result) != payload or read_manifest(result) != manifest:
        raise PackagingError("internal error: the embedded bundle does not round-trip")
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
    """Extracts the embedded directory (for inspection); returns the list of files."""
    payload = extract_bundle(Path(wasm_path).read_bytes())
    if payload is None:
        raise PackagingError(f"{wasm_path} carries no embedded bundle")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        for name in names:
            target = (output_directory / name).resolve()
            if output_directory.resolve() not in target.parents:
                raise PackagingError(f"refusing to extract {name!r} outside {output_directory}")
        archive.extractall(output_directory)
    return names
