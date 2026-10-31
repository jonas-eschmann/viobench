cmake_minimum_required(VERSION 3.20)
project(openvins C CXX)
# OpenVINS (github.com/rpng/open_vins, GPL-3.0) as a viobench solution: solution.cpp drives ov_msckf::VioManager (MSCKF, KLT
# tracking, static and dynamic initialization with Ceres). The sources are fetched at configure time (pinned, SHA256-checked).
# wasm (the submission): viobench/solution/runtime/python/build.sh (once, for OpenCV), then viobench/solution/runtime/native/build.sh <this directory>
# host debug build:      cmake -S . -B build-host && cmake --build build-host && build-host/debug <feed dir>   (needs OpenCV on the host)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(VIOBENCH_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/native" CACHE PATH "directory with viobench_abi.h and viobench_reactor.cmake")
include(${VIOBENCH_NATIVE_DIR}/viobench_reactor.cmake)

# fetch(<name> <url> <sha256> <top-level directory> [PATCH <file>] [PATHS <paths in the archive>...]): download the archive into
# build/downloads, unpack it (only PATHS, if given) into build/src/<name> and apply the patch; once per archive and patch
function(fetch name url sha256 top)
    cmake_parse_arguments(FETCH "" "PATCH" "PATHS" ${ARGN})
    set(directory ${CMAKE_BINARY_DIR}/src/${name})
    set(${name}_DIR ${directory} PARENT_SCOPE)
    set(stamp ${sha256})
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
        # outside any repository: the ceiling keeps git from treating a surrounding checkout as one; autocrlf=input
        # matches the LF patches against CRLF upstream files (e.g. ORB-SLAM3 Random.cpp) regardless of the user's git config
        execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}/src git -c core.autocrlf=input apply ${FETCH_PATCH}
                        WORKING_DIRECTORY ${directory} RESULT_VARIABLE result)
        if(result)
            message(FATAL_ERROR "${FETCH_PATCH} does not apply to ${directory}")
        endif()
    endif()
    file(WRITE ${directory}/.stamp ${stamp})
endfunction()

fetch(eigen https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz  # 3.4.0's JacobiSVD crashes on NaN input
      b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2 eigen-3.4.1 PATHS Eigen COPYING.MPL2)
fetch(ceres https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.tar.gz  # OpenVINS 2.7 needs LocalParameterization (gone in 2.2)
      ccbd716a93f65d4cb017e3090ae78809e02f5426dce16d0ee2b4f8a4ba2411a8 ceres-solver-2.1.0)
fetch(open_vins https://github.com/rpng/open_vins/archive/refs/tags/v2.7.tar.gz
      0135af4acd8dc099c8b50fc51c9faf53fe69462d0e6bad9e986ad4090af02df8 open_vins-2.7 PATCH ${CMAKE_CURRENT_SOURCE_DIR}/patches/open_vins.patch
      PATHS ov_core/src ov_init/src ov_msckf/src LICENSE)

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
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${eigen_DIR})
endif()
set(EIGEN3_INCLUDE_DIR ${eigen_DIR})
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

# Ceres, for dynamic initialization and fixed-lag smoothing: no threads, miniglog, Eigen for dense and sparse algebra, no optional backends
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
add_compile_definitions(MAX_LOG_LEVEL=-1)
add_subdirectory(${ceres_DIR} ${CMAKE_BINARY_DIR}/ceres EXCLUDE_FROM_ALL)

# OpenVINS: ov_core (tracking, features), ov_init (initializers), ov_msckf (the filter); no ROS, simulator or ArUco
set(OPENVINS_SOURCES
    ov_core/src/cpi/CpiV1.cpp ov_core/src/cpi/CpiV2.cpp
    ov_core/src/feat/Feature.cpp ov_core/src/feat/FeatureDatabase.cpp ov_core/src/feat/FeatureInitializer.cpp
    ov_core/src/track/TrackAruco.cpp ov_core/src/track/TrackBase.cpp ov_core/src/track/TrackDescriptor.cpp
    ov_core/src/track/TrackKLT.cpp ov_core/src/track/TrackSIM.cpp ov_core/src/types/Landmark.cpp ov_core/src/utils/print.cpp
    ov_init/src/ceres/Factor_GenericPrior.cpp ov_init/src/ceres/Factor_ImageReprojCalib.cpp ov_init/src/ceres/Factor_ImuCPIv1.cpp
    ov_init/src/ceres/State_JPLQuatLocal.cpp ov_init/src/dynamic/DynamicInitializer.cpp ov_init/src/init/InertialInitializer.cpp
    ov_init/src/static/StaticInitializer.cpp
    ov_msckf/src/core/VioManager.cpp ov_msckf/src/core/VioManagerHelper.cpp ov_msckf/src/state/Propagator.cpp
    ov_msckf/src/state/State.cpp ov_msckf/src/state/StateHelper.cpp ov_msckf/src/update/UpdaterHelper.cpp
    ov_msckf/src/update/UpdaterMSCKF.cpp ov_msckf/src/update/UpdaterSLAM.cpp ov_msckf/src/update/UpdaterZeroVelocity.cpp)
list(TRANSFORM OPENVINS_SOURCES PREPEND ${open_vins_DIR}/)
add_library(openvins STATIC ${OPENVINS_SOURCES})
# compat/ stands in for the three Boost headers OpenVINS includes (timing, chi-squared quantiles, filesystem)
target_include_directories(openvins PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/compat ${open_vins_DIR}/ov_core/src ${open_vins_DIR}/ov_init/src
                           ${open_vins_DIR}/ov_msckf/src)
target_compile_definitions(openvins PUBLIC ENABLE_ARUCO_TAGS=0 ROS_AVAILABLE=0)
target_compile_options(openvins PRIVATE -w)
target_link_libraries(openvins PUBLIC Ceres::ceres Eigen3::Eigen ${OpenCV_LIBS})

viobench_solution(solution solution.cpp)
target_link_libraries(solution PRIVATE openvins)
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    # wasm exceptions need libunwind; 8 MiB of stack below the data (an overflow traps instead of corrupting it)
    target_link_options(solution PRIVATE -fwasm-exceptions -Wl,-z,stack-size=8388608 -Wl,--stack-first)
    target_link_libraries(solution PRIVATE unwind)
endif()
