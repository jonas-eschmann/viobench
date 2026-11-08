# viobench_find_opencv(<components...>): host OpenCV for debugging, or the workspace's wasm build.
# Build missing wasm libraries with the shared Python runtime recipe, which also supplies OpenCV.
# A macro keeps find_package's variables in the caller's scope, just like a direct find_package call.
macro(viobench_find_opencv)
    if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
        get_filename_component(_viobench_workspace "${VIOBENCH_NATIVE_DIR}/../.." ABSOLUTE)
        set(_viobench_opencv_default "${_viobench_workspace}/build/runtime/opencv-build")
        set(VIOBENCH_OPENCV_DIR "${_viobench_opencv_default}" CACHE PATH "OpenCV built for wasm32-wasip1")
        set(_viobench_opencv_files "${VIOBENCH_OPENCV_DIR}/OpenCVConfig.cmake")
        set(_viobench_opencv_components ${ARGN})
        if(VIOBENCH_OPENCV_DIR STREQUAL _viobench_opencv_default)
            # The runtime's config imports all these libraries, including transitive dependencies.
            list(APPEND _viobench_opencv_components core imgproc features2d flann calib3d video dnn)
            list(APPEND _viobench_opencv_files "${VIOBENCH_OPENCV_DIR}/3rdparty/lib/liblibprotobuf.a"
                                               "${VIOBENCH_OPENCV_DIR}/3rdparty/lib/libzlib.a")
        endif()
        foreach(_viobench_component IN LISTS _viobench_opencv_components)
            list(APPEND _viobench_opencv_files "${VIOBENCH_OPENCV_DIR}/lib/libopencv_${_viobench_component}.a")
        endforeach()
        set(_viobench_opencv_missing FALSE)
        foreach(_viobench_file IN LISTS _viobench_opencv_files)
            if(NOT EXISTS "${_viobench_file}")
                set(_viobench_opencv_missing TRUE)
            endif()
        endforeach()
        if(_viobench_opencv_missing)
            if(NOT VIOBENCH_OPENCV_DIR STREQUAL _viobench_opencv_default)
                message(FATAL_ERROR "incomplete wasm OpenCV in ${VIOBENCH_OPENCV_DIR}: fix VIOBENCH_OPENCV_DIR or unset it to build the workspace's OpenCV automatically")
            endif()
            message(STATUS "viobench: building missing wasm OpenCV with the Python runtime (first build: 10-15 min)")
            execute_process(COMMAND "${CMAKE_COMMAND}" -E env "BUILD=${_viobench_workspace}/build/runtime"
                                    bash "${_viobench_workspace}/runtime/python/build.sh"
                            RESULT_VARIABLE _viobench_runtime_status)
            if(NOT _viobench_runtime_status STREQUAL "0")
                message(FATAL_ERROR "wasm OpenCV runtime build failed (${_viobench_runtime_status}); logs in ${_viobench_workspace}/build/runtime/logs")
            endif()
            foreach(_viobench_file IN LISTS _viobench_opencv_files)
                if(NOT EXISTS "${_viobench_file}")
                    message(FATAL_ERROR "runtime build did not produce ${_viobench_file}; logs in ${_viobench_workspace}/build/runtime/logs")
                endif()
            endforeach()
        endif()
        find_package(OpenCV REQUIRED COMPONENTS ${ARGN} PATHS "${VIOBENCH_OPENCV_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_package(OpenCV REQUIRED COMPONENTS ${ARGN})
    endif()
endmacro()

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
