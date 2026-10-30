"""Workspaces: solutions next to their own copy of the build recipe (toolchain/, runtime/), free to be changed.

`relpose init` creates one from this package: toolchain/ and runtime/, an example as solution/, and build/ for
everything built there (toolchain downloads, runtime.wasm). In a source checkout, relpose/solution is the workspace.
Commands find the workspace of a solution directory by walking up from it.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent  # relpose/solution
MARKER = Path("toolchain") / "download_dependencies.sh"
EXAMPLES = {"python": "python/minimal", "native": "native/cpp_minimal"}
EXAMPLE_NATIVE_DIR = "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/native"  # the example's CMakeLists.txt; solution/ is one level deep
IGNORED = shutil.ignore_patterns("__pycache__", "build", "build-host", "bundler")  # bundler/ is relpose's own code, not the recipe


class WorkspaceError(Exception):
    """No workspace, a failed build, or a directory that cannot be created."""


def find(directory):
    """The workspace containing directory, else this package's when it is a source checkout (pip install -e)."""
    directory = Path(directory).resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / MARKER).is_file():
            return candidate
    if (PACKAGE.parents[1] / "pyproject.toml").is_file():
        return PACKAGE
    raise WorkspaceError(f"{directory} is not inside a workspace: start one with `relpose init python <directory>` (or pass --runtime)")


def runtime_path(workspace):
    return Path(workspace) / "build" / "runtime" / "dist" / "runtime.wasm"


def run(workspace, script, *arguments):
    """Runs one of the workspace's build scripts."""
    environment = dict(os.environ)
    environment.setdefault("PYTHON3", sys.executable)  # new enough for the build, unlike macOS's /usr/bin/python3
    status = subprocess.run(["bash", str(Path(workspace) / script), *map(str, arguments)], env=environment).returncode
    if status:
        raise WorkspaceError(f"{Path(workspace) / script} failed (exit status {status})")


def build_runtime(workspace, force=False):
    """Builds the workspace's runtime.wasm (incrementally, from scratch with force) and returns its path."""
    build = Path(workspace) / "build" / "runtime"
    if force and build.exists():
        print(f"removing {build}")
        shutil.rmtree(build)
    print(f"building {runtime_path(workspace)}: the first build takes 10-15 min and downloads the toolchain into "
          f"{Path(workspace) / 'build' / 'toolchain'}", flush=True)
    try:
        run(workspace, "runtime/python/build.sh")
    except WorkspaceError as error:
        raise WorkspaceError(f"{error}; logs in {build / 'logs'}") from None
    if not runtime_path(workspace).is_file():
        raise WorkspaceError(f"the build did not write {runtime_path(workspace)} (is $BUILD set?)")
    return runtime_path(workspace)


def build_native(directory, output):
    """Builds a native solution directory (with a CMakeLists.txt) with its workspace's toolchain and copies the module to output."""
    directory = Path(directory)
    run(find(directory), "runtime/native/build.sh", directory)
    modules = sorted((directory / "build").glob("*.wasm"))
    if len(modules) != 1:
        raise WorkspaceError(f"expected one module in {directory / 'build'}, found {[module.name for module in modules]}")
    shutil.copyfile(modules[0], output)


def init(flavour, directory):
    """Creates a workspace with the selected example as solution/."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise WorkspaceError(f"{directory} exists and is not empty")
    for name in ("toolchain", "runtime"):
        shutil.copytree(PACKAGE / name, directory / name, ignore=IGNORED)
    shutil.copytree(PACKAGE / "examples" / EXAMPLES[flavour], directory / "solution", ignore=IGNORED)
    cmake = directory / "solution" / "CMakeLists.txt"
    if cmake.is_file():
        cmake.write_text(cmake.read_text().replace(EXAMPLE_NATIVE_DIR, "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/native"))
    (directory / ".gitignore").write_text("build/\nbuild-host/\n")
