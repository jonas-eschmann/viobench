cmake_minimum_required(VERSION 3.20)
project(vins_mono C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(VIOBENCH_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/native" CACHE PATH "viobench native runtime directory")
include(${VIOBENCH_NATIVE_DIR}/viobench_reactor.cmake)

# fetch(<name> <url> <sha256> <top-level directory> [PATCH <file>] [PATHS <paths in the archive>...]): download the archive into
# build/downloads, unpack it (only PATHS, if given) into build/src/<name> and apply the patch; once per archive and patch
function(fetch name url sha256 top)
    cmake_parse_arguments(FETCH "" "PATCH" "PATHS" ${ARGN})
    set(directory ${CMAKE_BINARY_DIR}/src/${name})
    set(${name}_DIR ${directory} PARENT_SCOPE)
    set(stamp "${sha256};${FETCH_PATHS}")
    if(FETCH_PATCH)
        file(SHA256 ${FETCH_PATCH} patch_sha256)
        string(APPEND stamp " ${patch_sha256}")
    endif()
    if(EXISTS ${directory}/.stamp)
        file(READ ${directory}/.stamp current)
        if(current STREQUAL stamp)
            return()
        endif()
    endif()
    set(archive ${CMAKE_BINARY_DIR}/downloads/${name}-${sha256}.tar.gz)
    message(STATUS "fetching ${url}")
    file(DOWNLOAD ${url} ${archive} EXPECTED_HASH SHA256=${sha256} STATUS status)
    list(GET status 0 code)
    if(code)
        message(FATAL_ERROR "download of ${url} failed: ${status}")
    endif()
    file(REMOVE_RECURSE ${directory} ${directory}.unpack)
    set(patterns)
    foreach(path ${FETCH_PATHS})
        list(APPEND patterns ${top}/${path})
    endforeach()
    if(patterns)
        file(ARCHIVE_EXTRACT INPUT ${archive} DESTINATION ${directory}.unpack PATTERNS ${patterns})
    else()
        file(ARCHIVE_EXTRACT INPUT ${archive} DESTINATION ${directory}.unpack)
    endif()
    file(RENAME ${directory}.unpack/${top} ${directory})
    file(REMOVE_RECURSE ${directory}.unpack)
    if(FETCH_PATCH)
        # outside any repository: the ceiling keeps git from treating a surrounding checkout as one
        # One diagnostic hunk omits trailing context containing upstream trailing spaces.
        execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}/src git apply --unidiff-zero ${FETCH_PATCH}
                        WORKING_DIRECTORY ${directory} RESULT_VARIABLE result)
        if(result)
            message(FATAL_ERROR "${FETCH_PATCH} does not apply to ${directory}")
        endif()
    endif()
    file(WRITE ${directory}/.stamp "${stamp}")
endfunction()

fetch(eigen https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
      b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2 eigen-3.4.1 PATHS Eigen COPYING.MPL2)
fetch(ceres https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.tar.gz
      ccbd716a93f65d4cb017e3090ae78809e02f5426dce16d0ee2b4f8a4ba2411a8 ceres-solver-2.1.0)
fetch(vins_mono https://github.com/HKUST-Aerial-Robotics/VINS-Mono/archive/90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d.tar.gz
      f0810285af9caeecc2b74a6b388d87c46165d664d80d5b5222f0738541c0f544 VINS-Mono-90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d
      PATCH ${CMAKE_CURRENT_SOURCE_DIR}/patches/vins_mono.patch PATHS vins_estimator/src LICENCE)

if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    # everything (Ceres included) links against the OpenCV of the Python runtime build, which uses wasm exceptions
    add_compile_options(-msimd128 -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false")
    get_filename_component(VIOBENCH_WORKSPACE ${VIOBENCH_NATIVE_DIR}/../.. ABSOLUTE)
    set(VIOBENCH_OPENCV_DIR ${VIOBENCH_WORKSPACE}/build/runtime/opencv-build CACHE PATH "OpenCV built for wasm32-wasip1 by viobench build-runtime")
    if(NOT EXISTS ${VIOBENCH_OPENCV_DIR}/OpenCVConfig.cmake)
        message(FATAL_ERROR "no wasm OpenCV in ${VIOBENCH_OPENCV_DIR}: run viobench build-runtime first")
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d video calib3d PATHS ${VIOBENCH_OPENCV_DIR} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d video calib3d)
endif()

# Eigen is header-only; Ceres finds it through this config
file(WRITE ${CMAKE_BINARY_DIR}/eigen3/Eigen3Config.cmake "
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${eigen_DIR}\")
endif()
set(EIGEN3_INCLUDE_DIR \"${eigen_DIR}\")
")
file(WRITE ${CMAKE_BINARY_DIR}/eigen3/Eigen3ConfigVersion.cmake "
set(PACKAGE_VERSION 3.4.1)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
if(PACKAGE_FIND_VERSION AND (PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION OR NOT PACKAGE_FIND_VERSION_MAJOR EQUAL 3))
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()
")
set(Eigen3_DIR ${CMAKE_BINARY_DIR}/eigen3 CACHE PATH "" FORCE)
find_package(Eigen3 REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

# Ceres, for initialization and sliding-window optimization: no threads, miniglog, Eigen for dense and sparse algebra, no optional backends
set(CERES_THREADING_MODEL NO_THREADS CACHE STRING "" FORCE)
foreach(option MINIGLOG EIGENSPARSE)
    set(${option} ON CACHE BOOL "" FORCE)
endforeach()
foreach(option GFLAGS SUITESPARSE CXSPARSE ACCELERATESPARSE CUDA LAPACK SCHUR_SPECIALIZATIONS BUILD_TESTING BUILD_EXAMPLES
               BUILD_BENCHMARKS BUILD_DOCUMENTATION BUILD_SHARED_LIBS PROVIDE_UNINSTALL_TARGET EXPORT_BUILD_DIR)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
# miniglog otherwise enables every VLOG level, including allocation diagnostics
# on every smoother update. Preserve warnings/errors without formatting verbose logs.
set(MINIGLOG_MAX_LOG_LEVEL -1 CACHE STRING "Only warnings and errors from miniglog" FORCE)
add_compile_definitions(MAX_LOG_LEVEL=-1)
add_subdirectory(${ceres_DIR} ${CMAKE_BINARY_DIR}/ceres EXCLUDE_FROM_ALL)

# The original estimator and factors run synchronously; the adapter supplies image tracks and calibration.
# ROS nodes, visualization, camera-model calibration tools and the separate pose graph are not built.
set(VINS_SOURCES estimator.cpp feature_manager.cpp
    factor/pose_local_parameterization.cpp factor/projection_factor.cpp factor/projection_td_factor.cpp
    factor/marginalization_factor.cpp utility/utility.cpp initial/solve_5pts.cpp initial/initial_aligment.cpp
    initial/initial_sfm.cpp initial/initial_ex_rotation.cpp)
list(TRANSFORM VINS_SOURCES PREPEND ${vins_mono_DIR}/vins_estimator/src/)
add_library(vins_mono STATIC ${VINS_SOURCES} parameters.cpp)
target_include_directories(vins_mono PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/compat ${vins_mono_DIR}/vins_estimator/src)
target_compile_definitions(vins_mono PUBLIC EIGEN_DONT_PARALLELIZE OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
target_compile_options(vins_mono PRIVATE -w)
target_link_libraries(vins_mono PUBLIC Ceres::ceres Eigen3::Eigen ${OpenCV_LIBS})

viobench_solution(solution solution.cpp)
target_link_libraries(solution PRIVATE vins_mono)
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    target_link_options(solution PRIVATE -fwasm-exceptions -Wl,-z,stack-size=8388608 -Wl,--stack-first)
    target_link_libraries(solution PRIVATE unwind)
endif()
