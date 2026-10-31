# viobench_solution(<target> <sources...>)  — one call, two builds:
#   * configured with the wasi-sdk toolchain file (viobench bundle does that):
#     the solution reactor <target>.wasm that you submit;
#   * configured with your host compiler (`cmake -S . -B build-host && cmake --build build-host`): an executable
#     `debug` = your sources + viobench_debug_main.c, built with -g, that reads a feed from `viobench feed` and calls your
#     init()/step() exactly like the sandbox — for gdb/lldb/VS Code.
function(viobench_solution target)
    set(sources ${ARGN})
    if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
        add_executable(${target} ${sources})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
        target_compile_options(${target} PRIVATE -O2 -msimd128)
        # a reactor: no main(), the VIOBENCH_EXPORT functions are the entry points; 64 MiB initial memory avoids early grows
        target_link_options(${target} PRIVATE -mexec-model=reactor -Wl,--initial-memory=67108864 -Wl,--strip-debug)
        set_target_properties(${target} PROPERTIES SUFFIX ".wasm")
    else()
        enable_language(C)
        add_executable(${target} ${sources} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/viobench_debug_main.c)
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
        target_compile_options(${target} PRIVATE -g -O1)
        set_target_properties(${target} PROPERTIES OUTPUT_NAME debug)
        message(STATUS "viobench: host configure -> building the `debug` driver (not a submission); use viobench bundle for the .wasm")
    endif()
endfunction()
