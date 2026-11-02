"""Exercise native bundling with real CMake and a small, offline runtime build recipe."""
import shutil
import subprocess

import pytest

from viobench.solution import workspace
from viobench.tools.cli import main as cli

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
MODULE = b"\0asm\x01\0\0\0"
OPENCV_CONFIG = "set(OpenCV_LIBS opencv_fixture)\nadd_library(opencv_fixture INTERFACE IMPORTED)\n"


@pytest.fixture
def native_workspace(tmp_path):
    root = tmp_path / "ws"
    workspace.init("native", root)
    # Avoid SDK downloads and compilation: the real shell/CMake pipeline copies a tiny module.
    (root / "toolchain/download_dependencies.sh").write_text("exit 0\n")
    toolchain = root / "build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("set(CMAKE_SYSTEM_NAME WASI)\n")
    (root / "solution/input.wasm").write_bytes(MODULE)
    (root / "solution/CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(opencv_prerequisite NONE)
include("${CMAKE_CURRENT_SOURCE_DIR}/../runtime/native/viobench_reactor.cmake")
viobench_find_opencv(core imgproc features2d video calib3d)
if(NOT OpenCV_LIBS STREQUAL "opencv_fixture" OR NOT TARGET opencv_fixture)
    message(FATAL_ERROR "OpenCV variables and targets must be available to the solution")
endif()
add_custom_target(solution ALL COMMAND "${CMAKE_COMMAND}" -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/input.wasm" "${CMAKE_CURRENT_BINARY_DIR}/solution.wasm")
''')
    (root / "runtime/python/build.sh").write_text('''
set -eu
echo "building fixture OpenCV"
mkdir -p "$BUILD/opencv-build/lib" "$BUILD/opencv-build/3rdparty/lib"
echo built >> "$BUILD/builds.txt"
for component in core imgproc features2d flann calib3d video dnn; do
    touch "$BUILD/opencv-build/lib/libopencv_$component.a"
done
touch "$BUILD/opencv-build/3rdparty/lib/libzlib.a" "$BUILD/opencv-build/3rdparty/lib/liblibprotobuf.a"
cat > "$BUILD/opencv-build/OpenCVConfig.cmake" <<'EOF'
''' + OPENCV_CONFIG + "EOF\n")
    return root


def test_native_bundle_builds_missing_opencv_and_reuses_it(native_workspace, tmp_path, capfd):
    root = native_workspace
    output = tmp_path / "solution.wasm"
    arguments = ["bundle", str(root / "solution"), "-o", str(output)]
    assert cli(arguments) == 0
    assert output.read_bytes() == MODULE
    assert "building fixture OpenCV" in capfd.readouterr().out
    assert "building fixture OpenCV" in (root / "solution/build/cmake.log").read_text()
    builds = root / "build/runtime/builds.txt"
    assert builds.read_text() == "built\n"

    assert cli(arguments) == 0
    assert builds.read_text() == "built\n"
    # A config file alone does not mean an interrupted runtime build finished its libraries.
    (root / "build/runtime/opencv-build/lib/libopencv_video.a").unlink()
    assert cli(arguments) == 0
    assert builds.read_text() == "built\nbuilt\n"


@pytest.mark.parametrize("script, message", [
    ("exit 7\n", "wasm OpenCV runtime build failed (7)"),
    ("exit 0\n", "runtime build did not produce"),
])
def test_native_bundle_stops_if_runtime_build_fails(native_workspace, tmp_path, script, message):
    root = native_workspace
    (root / "runtime/python/build.sh").write_text(script)
    output = tmp_path / "solution.wasm"
    output.write_bytes(b"previous submission")
    assert cli(["bundle", str(root / "solution"), "-o", str(output)]) == 1
    log = (root / "solution/build/cmake.log").read_text()
    assert message in log
    assert str(root / "build/runtime/logs") in log
    assert output.read_bytes() == b"previous submission"


@pytest.mark.parametrize("host", [False, True])
def test_existing_custom_or_host_opencv_skips_runtime(native_workspace, tmp_path, host):
    root = native_workspace
    opencv = tmp_path / "installed-opencv"
    (opencv / "lib").mkdir(parents=True)
    (opencv / "OpenCVConfig.cmake").write_text(OPENCV_CONFIG)
    for component in ("core", "imgproc", "features2d", "video", "calib3d"):
        (opencv / "lib" / f"libopencv_{component}.a").touch()
    arguments = [f"-DVIOBENCH_NATIVE_DIR={root / 'runtime/native'}"]
    arguments += [f"-DOpenCV_DIR={opencv}"] if host else ["-DCMAKE_SYSTEM_NAME=WASI", f"-DVIOBENCH_OPENCV_DIR={opencv}"]
    result = subprocess.run(["cmake", "-S", str(root / "solution"), "-B", str(tmp_path / "configure"), *arguments],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "build/runtime").exists()


def test_missing_custom_opencv_does_not_build_an_unrelated_runtime(native_workspace, tmp_path):
    root = native_workspace
    result = subprocess.run(["bash", str(root / "runtime/native/build.sh"), str(root / "solution"),
                             f"-DVIOBENCH_OPENCV_DIR={tmp_path / 'missing'}"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "fix VIOBENCH_OPENCV_DIR" in result.stdout
    assert not (root / "build/runtime").exists()
