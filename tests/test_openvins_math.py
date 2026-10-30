"""Native numerical contracts for the OpenVINS consensus output."""
from pathlib import Path
import shutil
import subprocess

import pytest

from relpose.solution import workspace


def test_consensus_output_contract(tmp_path):
    example = workspace.PACKAGE / "examples/native/openvins"
    eigen = example / "build/src/eigen"
    compiler = shutil.which("c++")
    if not compiler or not (eigen / "Eigen/Dense").is_file():
        pytest.skip("build the OpenVINS example and install a native C++ compiler for its numerical tests")

    executable = tmp_path / "openvins_consensus"
    source = Path(__file__).parent / "native/openvins_consensus.cpp"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         "-I", str(example), "-I", str(eigen), str(source), "-o", str(executable)],
        capture_output=True, text=True, check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    checked = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stdout + checked.stderr
