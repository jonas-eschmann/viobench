docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/work" -w /work \
    eecs116-wasm-python wasm-python/build.sh
