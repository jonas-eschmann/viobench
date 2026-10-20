# Toolchain for wasm-python/build.sh: wasi-sdk 34, Wasmtime 50, Cython, ninja, pytest and the pristine CPython 3.14.4, numpy 2.3.4 and OpenCV 4.12.0 sources (build.sh defaults point here)
FROM --platform=linux/amd64 ubuntu:26.04

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl git python3 python3-venv xz-utils && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/wasi-sdk.tar.gz https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sdk-34.0-x86_64-linux.tar.gz && echo "b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4  /tmp/wasi-sdk.tar.gz" | sha256sum -c - && tar -xzf /tmp/wasi-sdk.tar.gz -C /opt && mv /opt/wasi-sdk-34.0-x86_64-linux /opt/wasi-sdk && rm /tmp/wasi-sdk.tar.gz

RUN curl -fsSL -o /tmp/wasmtime.tar.xz https://github.com/bytecodealliance/wasmtime/releases/download/v50.0.0/wasmtime-v50.0.0-x86_64-linux.tar.xz && echo "77c19ee0437991f9ac7837b9c269d9c7076744f85dad263fea5214cc2723c7ff  /tmp/wasmtime.tar.xz" | sha256sum -c - && tar -xJf /tmp/wasmtime.tar.xz -C /opt && ln -s /opt/wasmtime-v50.0.0-x86_64-linux/wasmtime /usr/local/bin/wasmtime && rm /tmp/wasmtime.tar.xz

RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir cython==3.3.0 ninja==1.13.2 pytest==9.1.1 pluggy==1.6.0 iniconfig==2.3.0 packaging==26.3 pygments==2.20.0

RUN git clone --depth 1 --branch v3.14.4 https://github.com/python/cpython /opt/src/cpython && test "$(git -C /opt/src/cpython rev-parse HEAD)" = 23116f998f6789d8c2fbe5ed5b8146854c8c2a4f
RUN git clone --depth 1 --recursive --shallow-submodules --branch v2.3.4 https://github.com/numpy/numpy /opt/src/numpy && test "$(git -C /opt/src/numpy rev-parse HEAD)" = 1458b9e79d1a5755eae9adcb346758f449b6b430
RUN git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv /opt/src/opencv && test "$(git -C /opt/src/opencv rev-parse HEAD)" = 49486f61fb25722cbcf586b7f4320921d46fb38e

ENV PATH=/opt/venv/bin:$PATH
