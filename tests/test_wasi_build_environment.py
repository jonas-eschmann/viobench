"""Check the host/target environment boundary without compiling the runtime."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from viobench.solution import workspace


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_native_bootstrap_keeps_flags_and_wasi_build_clears_them(tmp_path):
    toolchain = tmp_path / "toolchain"
    for name in ("cpython", "numpy", "opencv"):
        (toolchain / "src" / name).mkdir(parents=True)
    tools = toolchain / "venv/bin"
    tools.mkdir(parents=True)
    (tools / "python3").symlink_to(sys.executable)
    # Patching and CPU affinity are outside this test's host/target boundary.
    for name, body in {"git": "exit 0", "taskset": 'shift 2\nexec "$@"'}.items():
        command = tools / name
        command.write_text("#!/bin/sh\n" + body + "\n")
        command.chmod(0o755)

    flags = {
        "CFLAGS": "-march=x86-64-v2",
        "CXXFLAGS": "-mtune=haswell",
        "CPPFLAGS": "-I/native/include",
        "LDFLAGS": "-Wl,--as-needed -L/native/lib",
        "CPATH": "/native/include",
        "C_INCLUDE_PATH": "/native/c-include",
        "CPLUS_INCLUDE_PATH": "/native/cxx-include",
        "LIBRARY_PATH": "/native/lib",
    }
    compilers = {"CC": "/native/bin/cc", "CXX": "/native/bin/c++"}
    host_environment = flags | compilers
    driver = toolchain / "src/cpython/Tools/wasm/wasi/__main__.py"
    driver.parent.mkdir(parents=True)
    driver.write_text(f'''
import json, os, sys
with open(os.environ["VIOBENCH_BUILD_STAGE_LOG"], "a") as log:
    json.dump({{"stage": sys.argv[1],
               "env": {{key: os.environ.get(key) for key in {tuple(host_environment)!r}}}}}, log)
    log.write("\\n")
# Stop at the WASI boundary; the remainder of the recipe needs real artifacts.
if sys.argv[1] != "build-python":
    sys.exit(23)
''')
    report = tmp_path / "stages.jsonl"
    env = dict(os.environ)
    env.update(host_environment)
    env.update({
        "TOOLCHAIN": str(toolchain),
        "SRC": str(toolchain / "src"),
        "VENV": str(tools.parent),
        "WASI_SDK_PATH": str(toolchain / "wasi-sdk"),
        "WASMTIME": str(toolchain / "wasmtime/wasmtime"),
        "BUILD": str(tmp_path / "build"),
        "VIOBENCH_BUILD_STAGE_LOG": str(report),
    })
    result = subprocess.run(
        ["bash", str(workspace.PACKAGE / "runtime/python/cpython/build-wasm.sh")],
        env=env, capture_output=True, text=True,
    )

    assert result.returncode == 23, result.stdout + result.stderr
    stages = [json.loads(line) for line in report.read_text().splitlines()]
    assert [entry["stage"] for entry in stages] == ["build-python", "build-host"]
    assert stages[0]["env"] == host_environment
    assert stages[1]["env"] == dict.fromkeys(flags) | compilers
