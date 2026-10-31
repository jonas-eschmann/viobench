#!/usr/bin/env bash
# CPython 3.14 + numpy + OpenCV (core, imgproc, features2d, flann, calib3d, video, dnn, cv2) statically linked into one self-contained wasm32-wasip1 python.wasm (stdlib and numpy Python code frozen in)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# toolchain/download_dependencies.sh puts these into the workspace's build/toolchain
TOOLCHAIN=${TOOLCHAIN:-$HERE/../../../build/toolchain}
SRC=${SRC:-$TOOLCHAIN/src}
SDK=${WASI_SDK_PATH:-$TOOLCHAIN/wasi-sdk}
VENV=${VENV:-$TOOLCHAIN/venv}
WASMTIME=${WASMTIME:-$TOOLCHAIN/wasmtime/wasmtime}
BUILD=${BUILD:-$HERE/../../../build/runtime}
CPUS=$(getconf _NPROCESSORS_ONLN)
JOBS=$(( CPUS < 5 ? CPUS : 5 ))
export PATH=$VENV/bin:$(dirname "$WASMTIME"):$PATH
mkdir -p "$BUILD/logs" "$BUILD/dist" "$BUILD/src"
BUILD=$(cd "$BUILD" && pwd)
case "$BUILD:$SDK:$VENV" in
    *" "*) echo "CPython's build cannot handle spaces in paths: move the workspace (or point BUILD/TOOLCHAIN) somewhere without them" >&2; exit 1 ;;
esac
# the source copies are unpacked tarballs, not repositories: keep git (the patches below, numpy's version script) from finding a repository around them
export GIT_CEILING_DIRECTORIES=$BUILD/src

# CPython builds in-tree and numpy/OpenCV get patched, so the build works on copies of the pristine sources
for project in cpython numpy opencv; do
    if [ ! -d "$BUILD/src/$project" ]; then
        cp -R "$SRC/$project" "$BUILD/src/$project"
    fi
done
for project in numpy opencv; do
    if ! git -C "$BUILD/src/$project" apply --check -R "$HERE/patches/$project.patch" 2>/dev/null; then
        git -C "$BUILD/src/$project" apply "$HERE/patches/$project.patch"
    fi
done

CPYTHON=$BUILD/src/cpython
# configured trees remember absolute paths: one from the former Docker toolchain (/work) or a moved workspace cannot be reused
if [ -f "$CPYTHON/cross-build/wasm32-wasip1/Makefile" ] && ! grep -qF "abs_srcdir=	$CPYTHON/" "$CPYTHON/cross-build/wasm32-wasip1/Makefile"; then
    echo "$BUILD was configured at another path (e.g. inside the former Docker toolchain): delete it and build again" >&2
    exit 1
fi
if [ ! -f "$CPYTHON/cross-build/wasm32-wasip1/python.wasm" ]; then
    PIN=""  # macOS has no taskset
    if command -v taskset > /dev/null; then
        PIN="taskset -c 0-$((JOBS - 1))"
    fi
    (cd "$CPYTHON" && WASI_SDK_PATH=$SDK $PIN python3 Tools/wasm/wasi build --quiet --logdir "$BUILD/logs")
fi
BUILD_PYTHON=$CPYTHON/cross-build/$(python3 -c "import sysconfig; print(sysconfig.get_config_var('BUILD_GNU_TYPE'))")/python
if [ ! -f "$BUILD_PYTHON" ]; then
    BUILD_PYTHON=$BUILD_PYTHON.exe  # case-insensitive file systems (macOS): python would clash with the Python/ directory
fi
make -C "$CPYTHON/cross-build/wasm32-wasip1" inclinstall libinstall DESTDIR="$BUILD/prefix" > "$BUILD/logs/install.log" 2>&1
mkdir -p "$BUILD/prefix/usr/local/lib/python3.14/lib-dynload"
make -s -C "$CPYTHON/cross-build/wasm32-wasip1" -n -W Programs/python.o python.wasm | grep -E '^[^[:space:]]*clang.*-o python.wasm' > "$BUILD/python-link-command.txt"

NUMPY=$BUILD/numpy-cross
mkdir -p "$NUMPY/sysconfig"
sed "s#'/usr/local/include/python3.14'#'$BUILD/prefix/usr/local/include/python3.14'#g" "$CPYTHON/cross-build/wasm32-wasip1/build/lib.wasi-wasm32-3.14/_sysconfigdata__wasi_wasm32-wasi.py" > "$NUMPY/sysconfig/_sysconfigdata__wasi_wasm32-wasi.py"
cat > "$NUMPY/python-target" <<EOF
#!/bin/sh
export _PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__wasi_wasm32-wasi
export _PYTHON_HOST_PLATFORM=wasi-wasm32
export PYTHONPATH=$NUMPY/sysconfig
exec $BUILD_PYTHON "\$@"
EOF
chmod +x "$NUMPY/python-target"
cat > "$NUMPY/cross.ini" <<EOF
[binaries]
c = '$SDK/bin/clang'
cpp = '$SDK/bin/clang++'
ar = '$SDK/bin/llvm-ar'
strip = '$SDK/bin/llvm-strip'
python = '$NUMPY/python-target'
exe_wrapper = ['$WASMTIME', 'run', '--dir=/']

[built-in options]
c_args = ['--target=wasm32-wasip1']
cpp_args = ['--target=wasm32-wasip1', '-fwasm-exceptions', '-mllvm', '-wasm-use-legacy-eh=false']
c_link_args = ['--target=wasm32-wasip1']
cpp_link_args = ['--target=wasm32-wasip1', '-fwasm-exceptions']

[host_machine]
system = 'wasi'
cpu_family = 'wasm32'
cpu = 'wasm32'
endian = 'little'
EOF
MESON="$BUILD/src/numpy/vendored-meson/meson/meson.py"
if [ ! -f "$NUMPY/build/build.ninja" ]; then
    python "$MESON" setup --cross-file "$NUMPY/cross.ini" "$NUMPY/build" "$BUILD/src/numpy" --buildtype=release -Db_ndebug=if-release -Dblas=none -Dlapack=none -Dallow-noblas=true -Ddisable-threading=true -Ddisable-svml=true -Ddisable-highway=true -Ddisable-intel-sort=true -Dcpu-baseline=none -Dcpu-dispatch=none > "$BUILD/logs/numpy-setup.log"
fi
# extension modules are shared libraries in meson, which wasm32-wasip1 cannot link: their objects are archived below instead
ninja -C "$NUMPY/build" -j "$JOBS" -k 0 > "$BUILD/logs/numpy-build.log" 2>&1 || true
rm -rf "$NUMPY/install"
python "$MESON" install -C "$NUMPY/build" --destdir ../install --tags python-runtime,devel --no-rebuild > "$BUILD/logs/numpy-install.log"
python "$HERE/scripts/collect_numpy.py" "$NUMPY/build" "$NUMPY/archives" "$SDK/bin/llvm-ar" "$MESON"
rm -rf "$BUILD/prefix/usr/local/lib/python3.14/site-packages/numpy"
cp -r "$NUMPY/install/usr/local/lib/python3.14/site-packages/numpy" "$BUILD/prefix/usr/local/lib/python3.14/site-packages/"

OPENCV=$BUILD/opencv-build
if [ ! -f "$OPENCV/build.ninja" ]; then
    DISABLED=""
    for feature in PTHREADS_PF OPENCL IPP ITT LAPACK EIGEN ADE JPEG PNG SPNG TIFF WEBP OPENJPEG JASPER OPENEXR AVIF FFMPEG GSTREAMER GTK QT VTK V4L 1394 TBB OPENMP CAROTENE KLEIDICV FLATBUFFERS VA VA_INTEL QUIRC OBSENSOR CUDA GDAL GDCM OPENVX OPENNI2 LIBREALSENSE WAYLAND IMGCODEC_HDR IMGCODEC_SUNRASTER IMGCODEC_PXM IMGCODEC_PFM HAL_RVV NDSRVP; do
        DISABLED="$DISABLED -DWITH_$feature=OFF"
    done
    WASI_DEFINES="-D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID"
    # OPENCV_DISABLE_THREAD_SUPPORT turns the python3 and dnn modules off unless BUILD_opencv_python3/BUILD_opencv_dnn are set explicitly;
    # dnn (ONNX inference via cv2.dnn) needs the bundled protobuf, which builds fine on wasi-libc's single-threaded pthread stubs
    cmake -S "$BUILD/src/opencv" -B "$OPENCV" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$SDK/share/cmake/wasi-sdk-p1.cmake" -DWASI_SDK_PREFIX="$SDK" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CROSSCOMPILING_EMULATOR=$WASMTIME;run;--dir=/" "-DCMAKE_C_FLAGS=-msimd128 $WASI_DEFINES" "-DCMAKE_CXX_FLAGS=-msimd128 -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false $WASI_DEFINES" -DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc,features2d,flann,calib3d,video,dnn,python3 -DOPENCV_DISABLE_THREAD_SUPPORT=ON -DBUILD_opencv_python3=ON -DBUILD_opencv_dnn=ON -DWITH_PROTOBUF=ON -DBUILD_PROTOBUF=ON -DPROTOBUF_UPDATE_FILES=OFF -DOPENCV_DNN_OPENCL=OFF -DOPENCV_DNN_CUDA=OFF -DOPENCV_DNN_TFLITE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_js=OFF -DOPENCV_ENABLE_NONFREE=OFF $DISABLED -DOPENCV_SKIP_PYTHON_LOADER=ON -DPYTHON3_EXECUTABLE="$VENV/bin/python" -DPYTHON3_INCLUDE_DIR="$BUILD/prefix/usr/local/include/python3.14" -DPYTHON3_LIBRARY="$CPYTHON/cross-build/wasm32-wasip1/libpython3.14.a" -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY/install/usr/local/lib/python3.14/site-packages/numpy/_core/include" -DPYTHON3_PACKAGES_PATH=lib/python3.14/site-packages > "$BUILD/logs/opencv-configure.log"
fi
# the cv2 module target is a shared library as well: only its objects are used
ninja -C "$OPENCV" -j "$JOBS" -k 0 > "$BUILD/logs/opencv-build.log" 2>&1 || true
mkdir -p "$BUILD/opencv-archives"
rm -f "$BUILD/opencv-archives/libcv2.a"
"$SDK/bin/llvm-ar" rcs "$BUILD/opencv-archives/libcv2.a" "$OPENCV"/modules/python3/CMakeFiles/opencv_python3.dir/__/src2/{cv2,cv2_util,cv2_numpy,cv2_convert,cv2_highgui}.cpp.obj
python - "$BUILD" <<'EOF'
import json, pathlib, sys
build = pathlib.Path(sys.argv[1])
opencv = build / "opencv-build"
libraries = [opencv / f"lib/libopencv_{name}.a" for name in ["dnn", "video", "calib3d", "features2d", "flann", "imgproc", "core"]] + [opencv / "3rdparty/lib/liblibprotobuf.a", opencv / "3rdparty/lib/libzlib.a"]
missing = [str(library) for library in libraries if not library.exists()]
if missing:
    sys.exit(f"missing OpenCV libraries (see logs/opencv-build.log): {missing}")
manifest = {"modules": [{"name": "cv2", "init": "PyInit_cv2", "archive": str(build / "opencv-archives/libcv2.a")}], "static_libraries": [str(library) for library in libraries]}
(build / "opencv-archives/modules.json").write_text(json.dumps(manifest, indent=1))
EOF

# frozen modules get __file__ under sys._stdlib_dir, so numpy's bytecode is named after /usr/local/lib/python3.14/numpy as well
LIB=$BUILD/prefix/usr/local/lib/python3.14
# viobench/solution/runtime/python/guest holds the frozen guest-side shim (viobench_guest) of the solution runtime
"$BUILD_PYTHON" "$HERE/scripts/freeze.py" "$BUILD/frozen" "$LIB:/usr/local/lib/python3.14:" "$LIB/site-packages/numpy:/usr/local/lib/python3.14/numpy:numpy" "$HERE/../guest:/usr/local/lib/python3.14:"
python "$HERE/scripts/link.py" "$BUILD" "$SDK" "$BUILD/dist/python.wasm" "$NUMPY/archives/modules.json" "$BUILD/opencv-archives/modules.json" --frozen "$BUILD/frozen"
"$WASMTIME" compile "$BUILD/dist/python.wasm" -o "$BUILD/dist/python.cwasm"
"$WASMTIME" run --allow-precompiled "$BUILD/dist/python.cwasm" - < "$HERE/smoke_vo.py"
# runtime.wasm: the same interpreter as a reactor exporting the solution ABI (docs/ABI.md); students bundle their code into it
python "$HERE/scripts/link.py" "$BUILD" "$SDK" "$BUILD/dist/runtime.wasm" "$NUMPY/archives/modules.json" "$BUILD/opencv-archives/modules.json" --frozen "$BUILD/frozen" --reactor "$HERE/../reactor.c"
