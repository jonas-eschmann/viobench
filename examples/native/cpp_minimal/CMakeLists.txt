cmake_minimum_required(VERSION 3.20)
project(solution CXX)
# configure with the wasi-sdk toolchain file:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/opt/wasi-sdk/share/cmake/wasi-sdk-p1.cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_executable(solution solution.cpp)
target_compile_options(solution PRIVATE -O2 -msimd128 -fno-exceptions)
# reactor: no main(), the exports in solution.cpp are the entry points; 64 MiB initial memory avoids early grows
target_link_options(solution PRIVATE -mexec-model=reactor -Wl,--initial-memory=67108864 -Wl,--strip-debug)
set_target_properties(solution PROPERTIES SUFFIX ".wasm")
