# Include after add_executable(<target> ...) to turn it into a relpose solution module:
#   include(${RELPOSE_NATIVE_DIR}/relpose_reactor.cmake)
#   relpose_solution(solution)
# Configure with the wasi-sdk toolchain file (the toolchain image has it at /opt/wasi-sdk):
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/opt/wasi-sdk/share/cmake/wasi-sdk-p1.cmake
if(NOT CMAKE_SYSTEM_NAME STREQUAL "WASI")
    message(FATAL_ERROR "relpose: configure with -DCMAKE_TOOLCHAIN_FILE=<wasi-sdk>/share/cmake/wasi-sdk-p1.cmake (see submission/native/build.sh)")
endif()

function(relpose_solution target)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
    target_compile_options(${target} PRIVATE -O2 -msimd128)
    # a reactor: no main(), the RELPOSE_EXPORT functions are the entry points; 64 MiB initial memory avoids early grows
    target_link_options(${target} PRIVATE -mexec-model=reactor -Wl,--initial-memory=67108864 -Wl,--strip-debug)
    set_target_properties(${target} PROPERTIES SUFFIX ".wasm")
endfunction()
