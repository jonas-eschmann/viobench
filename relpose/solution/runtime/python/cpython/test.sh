#!/usr/bin/env bash
# runs numpy's own test files inside dist/python.cwasm, pytest taken from $VENV; arguments are paths relative to the numpy package
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HERE/../../../build/runtime}
TOOLCHAIN=${TOOLCHAIN:-$HERE/../../../build/toolchain}
VENV=${VENV:-$TOOLCHAIN/venv}
WASMTIME=${WASMTIME:-$TOOLCHAIN/wasmtime/wasmtime}
TESTS=$BUILD/tests
SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then
    SUITES=(linalg/tests/test_linalg.py fft/tests random/tests/test_randomstate.py random/tests/test_generator_mt19937.py random/tests/test_random.py)
fi

rm -rf "$TESTS"
mkdir -p "$TESTS/site-packages"
SITE=$("$VENV/bin/python" -c "import os, pytest; print(os.path.dirname(os.path.dirname(pytest.__file__)))")
for package in pytest _pytest pluggy iniconfig packaging pygments py.py; do
    cp -r "$SITE/$package" "$TESTS/site-packages/"
done
# empty parent __init__.py files make pytest name the modules numpy.fft.tests.test_pocketfft instead of colliding tests.* packages; the frozen numpy still wins
(cd "$BUILD/src/numpy" && find numpy -type d -name tests) | while read -r directory; do
    mkdir -p "$TESTS/$directory"
    cp -r "$BUILD/src/numpy/$directory/." "$TESTS/$directory/"
    parent=$(dirname "$directory")
    while [ "$parent" != "." ]; do
        touch "$TESTS/$parent/__init__.py"
        parent=$(dirname "$parent")
    done
done

# numpy itself is frozen into the binary; its frozen packages' __path__ points at /usr/local/lib/python3.14/numpy, where the test files are mounted
PATHS=()
for suite in "${SUITES[@]}"; do
    PATHS+=("/usr/local/lib/python3.14/numpy/$suite")
done
# WASI has no dup() and no /dev/null (capture, faulthandler and logging plugins); test_legacy_pickle needs zlib
"$WASMTIME" run --allow-precompiled --dir "$TESTS::/" --dir "$TESTS/numpy::/usr/local/lib/python3.14/numpy" --env PYTHONPATH=/site-packages --env HOME=/ "$BUILD/dist/python.cwasm" -m pytest -q -p no:cacheprovider --noconftest --capture=sys -p no:faulthandler -p no:logging -W ignore::pytest.PytestUnknownMarkWarning -k "not test_legacy_pickle" "${PATHS[@]}"
