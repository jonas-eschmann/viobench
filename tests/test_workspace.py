"""Workspaces (`viobench init`): a solution next to its own copy of the build recipe, which the commands find from inside."""
import shutil
import subprocess

import pytest

from viobench.solution import workspace
from viobench.solution.runtime.python.bundler import BundleError
from viobench.solution.runtime.python.bundler.bundle import bundle
from viobench.tools.cli import main as cli


def test_init_copies_the_recipe_and_the_minimal_example(tmp_path):
    root = tmp_path / "ws"
    assert cli(["init", "python", str(root)]) == 0
    assert (root / "solution" / "solution.py").read_text() == (workspace.PACKAGE / "examples/python/minimal/solution.py").read_text()
    for name in ("toolchain/download_dependencies.sh", "runtime/python/build.sh", "runtime/python/cpython/patches/numpy.patch", "runtime/native/viobench_abi.h"):
        assert (root / name).is_file(), name
    assert not (root / "runtime" / "python" / "bundler").exists()
    assert workspace.find(root / "solution") == root.resolve()
    with pytest.raises(BundleError, match="is a workspace"):
        bundle(root, tmp_path / "x.wasm")
    assert cli(["init", "python", str(root)]) == 1  # not empty


def test_outside_a_workspace_only_a_checkout_falls_back_to_its_own(tmp_path, monkeypatch):
    assert workspace.find(tmp_path) == workspace.PACKAGE
    monkeypatch.setattr(workspace, "PACKAGE", tmp_path / "site-packages" / "viobench" / "solution")
    with pytest.raises(workspace.WorkspaceError, match="viobench init"):
        workspace.find(tmp_path)


def test_bundle_uses_the_runtime_of_its_workspace(runtime_path, tmp_path, monkeypatch):
    monkeypatch.delenv("VIOBENCH_RUNTIME", raising=False)
    root = tmp_path / "ws"
    workspace.init("python", root)
    target = workspace.runtime_path(root)
    target.parent.mkdir(parents=True)
    shutil.copy(runtime_path, target)
    assert bundle(root / "solution", tmp_path / "x.wasm")["runtime"] == str(target)


@pytest.fixture
def runtime_build_recipe(tmp_path):
    root = tmp_path / "ws"
    script = root / "runtime/python/build.sh"
    script.parent.mkdir(parents=True)
    script.write_text('''
set -eu
mkdir -p "$BUILD/dist" "$BUILD/logs"
printf 'fixture runtime' > "$BUILD/dist/runtime.wasm"
printf 'build completed' > "$BUILD/logs/build.log"
''')
    return root


def test_runtime_build_ignores_conda_build_triplet(runtime_build_recipe, tmp_path, monkeypatch):
    root = runtime_build_recipe
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BUILD", "x86_64-conda-linux-gnu")

    target = workspace.build_runtime(root)

    assert target == root / "build/runtime/dist/runtime.wasm"
    assert target.is_absolute()
    assert target.read_text() == "fixture runtime"
    assert (root / "build/runtime/logs/build.log").read_text() == "build completed"
    assert not (tmp_path / "x86_64-conda-linux-gnu").exists()


def test_direct_recipe_run_preserves_custom_build_directory(runtime_build_recipe, tmp_path, monkeypatch):
    build = tmp_path / "custom-runtime"
    monkeypatch.setenv("BUILD", str(build))

    workspace.run(runtime_build_recipe, "runtime/python/build.sh")

    assert (build / "dist/runtime.wasm").read_text() == "fixture runtime"
    assert (build / "logs/build.log").read_text() == "build completed"
    assert not (runtime_build_recipe / "build").exists()


@pytest.mark.skipif(shutil.which("cmake") is None or (shutil.which("cc") is None and shutil.which("gcc") is None and shutil.which("clang") is None),
                    reason="needs cmake and a host C/C++ compiler")
def test_native_workspace_finds_its_runtime_directory(tmp_path):
    root = tmp_path / "ws"
    workspace.init("native", root)
    build = tmp_path / "build-host"
    configure = subprocess.run(["cmake", "-S", str(root / "solution"), "-B", str(build)], capture_output=True, text=True)
    assert configure.returncode == 0, configure.stdout + configure.stderr
    compile_ = subprocess.run(["cmake", "--build", str(build)], capture_output=True, text=True)
    assert compile_.returncode == 0, compile_.stdout + compile_.stderr
    assert (build / "debug").is_file()
