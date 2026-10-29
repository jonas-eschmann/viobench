"""`relpose bundle`: a directory of Python code (plus data files) → one self-contained solution.wasm; `unbundle` reverses it.

The directory is zipped (uncompressed, reproducible) and embedded into the Python runtime as a data segment
(`embed.py`). Nothing is compiled, so no toolchain is needed beyond the runtime itself.
"""
import ast
import hashlib
import io
import os
import zipfile
from pathlib import Path

from relpose.solution import workspace
from relpose.tools import abi
from relpose.tools.abi.module import embedded_bundle

from . import BundleError as PackagingError
from .embed import embed_bundle, strip_bundle

IGNORED_DIRECTORIES = {"__pycache__", "venv", "node_modules"}  # plus every directory starting with "."
IGNORED_SUFFIXES = {".pyc", ".pyo"}
SIZE_WARNING_BYTES = 64 << 20
RUNTIME_VARIABLE = "RELPOSE_RUNTIME"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)



def default_runtime(directory):
    """runtime.wasm from $RELPOSE_RUNTIME, else from the workspace of directory, built first if missing."""
    if os.environ.get(RUNTIME_VARIABLE):
        path = Path(os.environ[RUNTIME_VARIABLE])
        if not path.is_file():
            raise PackagingError(f"${RUNTIME_VARIABLE} is {path}, which does not exist")
        return path
    space = workspace.find(directory)
    path = workspace.runtime_path(space)
    return path if path.is_file() else workspace.build_runtime(space)


def collect_files(directory):
    """Relative paths of everything that goes into the bundle, sorted; refuses symlinks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PackagingError(f"{directory} is not a directory")
    if (directory / workspace.MARKER).is_file():
        raise PackagingError(f"{directory} is a workspace, not a solution: bundle its solution directory")
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


def validate(directory, files):
    """Syntax-checks every .py and looks for class Solution in solution.py; returns warnings (hard errors raise)."""
    directory = Path(directory)
    module_name, class_name = abi.SOLUTION_MODULE, abi.SOLUTION_CLASS
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
        raise PackagingError(f"no {module_name}.py or {module_name}/__init__.py in {directory}: the bundle's entry point is class {class_name} there")
    classes = {node.name: node for node in ast.walk(trees[main]) if isinstance(node, ast.ClassDef)}
    imported = {alias.asname or alias.name for node in trees[main].body if isinstance(node, ast.ImportFrom) for alias in node.names}
    if class_name in imported:
        pass  # re-exported from another module (`from mypkg import Estimator as Solution`): checked when it is imported
    elif class_name not in classes:
        warnings.append(f"{main} neither defines nor imports {class_name} (fine if it binds it some other way)")
    else:
        methods = {node.name for node in classes[class_name].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for method in ("__init__", "step"):
            if method not in methods:
                warnings.append(f"class {class_name} in {main} has no {method} method (fine if inherited)")
    total = sum((directory / file).stat().st_size for file in files)
    if total > SIZE_WARNING_BYTES:
        warnings.append(f"bundle is {total / 2**20:.0f} MiB; the submission limit may be lower")
    return warnings


def make_zip(directory, files):
    """Uncompressed (the guest has no zlib), deterministic zip."""
    directory = Path(directory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            info = zipfile.ZipInfo(file.as_posix(), date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (directory / file).read_bytes())
    return buffer.getvalue()


def bundle(directory, output, runtime=None):
    """Writes output (.wasm) and returns a summary dict."""
    directory = Path(directory)
    files = collect_files(directory)
    warnings = validate(directory, files)
    runtime_path = Path(runtime) if runtime else default_runtime(directory)
    runtime_bytes = runtime_path.read_bytes()
    payload = make_zip(directory, files)
    result = embed_bundle(runtime_bytes, payload)
    if embedded_bundle(result) != payload:
        raise PackagingError("internal error: the embedded bundle does not round-trip")
    output = Path(output)
    output.write_bytes(result)
    return {
        "output": str(output),
        "bytes": len(result),
        "files": [file.as_posix() for file in files],
        "bundle_bytes": len(payload),
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(strip_bundle(runtime_bytes)).hexdigest(),
        "warnings": warnings,
    }


def unbundle(wasm_path, output_directory):
    """Extracts the embedded directory (for inspection); returns the list of files."""
    payload = embedded_bundle(Path(wasm_path).read_bytes())
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
