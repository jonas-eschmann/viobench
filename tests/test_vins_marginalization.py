"""Numerical invariants of the production VINS square-root marginalizer."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from relpose.solution import workspace


def test_square_root_marginalization(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("needs a host C++ compiler")
    example = workspace.PACKAGE / "examples/native/vins_mono"
    eigen = Path(os.environ.get("RELPOSE_EIGEN") or example / "build/src/eigen")
    if not (eigen / "Eigen/Dense").is_file():
        pytest.skip("build VINS-Mono first or set RELPOSE_EIGEN to Eigen headers")
    source = Path(__file__).parent / "native/vins_marginalization.cpp"
    executable = tmp_path / "vins_marginalization"
    build = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(eigen), "-I", str(example / "compat"),
                            str(source), "-o", str(executable)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
