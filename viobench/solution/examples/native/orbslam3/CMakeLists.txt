cmake_minimum_required(VERSION 3.20)
project(orbslam3 C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(VIOBENCH_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/native" CACHE PATH "viobench native runtime directory")
include(${VIOBENCH_NATIVE_DIR}/viobench_reactor.cmake)
viobench_find_opencv(core imgproc features2d calib3d)
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    add_compile_options(-msimd128 -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false")
endif()

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
        # outside any repository: the ceiling keeps git from treating a surrounding checkout as one; autocrlf=input
        # matches the LF patches against CRLF upstream files (e.g. ORB-SLAM3 Random.cpp) regardless of the user's git config
        execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}/src git -c core.autocrlf=input apply ${FETCH_PATCH}
                        WORKING_DIRECTORY ${directory} RESULT_VARIABLE result)
        if(result)
            message(FATAL_ERROR "${FETCH_PATCH} does not apply to ${directory}")
        endif()
    endif()
    file(WRITE ${directory}/.stamp "${stamp}")
endfunction()

fetch(eigen https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
      b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2 eigen-3.4.1 PATHS Eigen unsupported COPYING.MPL2)
fetch(orbslam3 https://github.com/UZ-SLAMLab/ORB_SLAM3/archive/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4.tar.gz
      b69fe1809588a8610580db1d5e6b77c63b5e4b5b47abc16ed839e93fbc7e7ddf ORB_SLAM3-4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
      PATCH ${CMAKE_CURRENT_SOURCE_DIR}/patches/orbslam3.patch
      PATHS include src Thirdparty/DBoW2 Thirdparty/g2o Thirdparty/Sophus Vocabulary LICENSE Dependencies.md)
# Headless system/viewer adapters and a database-only loop closer; estimation uses the upstream local mapper.
foreach(header System FrameDrawer MapDrawer Viewer LoopClosing)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compat/${header}.h ${orbslam3_DIR}/include/${header}.h COPYONLY)
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

# Bundled g2o and DBoW2 are built directly: no upstream shared libraries or -march=native.
set(G2O_DIR ${orbslam3_DIR}/Thirdparty/g2o)
configure_file(${G2O_DIR}/config.h.in ${G2O_DIR}/config.h)
file(GLOB G2O_SOURCES CONFIGURE_DEPENDS ${G2O_DIR}/g2o/core/*.cpp ${G2O_DIR}/g2o/types/types_*.cpp ${G2O_DIR}/g2o/stuff/*.cpp)
list(FILTER G2O_SOURCES EXCLUDE REGEX "_test\\.cpp$")
add_library(g2o STATIC ${G2O_SOURCES} ${G2O_DIR}/g2o/stuff/os_specific.c)
target_include_directories(g2o PUBLIC ${eigen_DIR} ${G2O_DIR})
target_compile_options(g2o PRIVATE -w)

set(DBOW2_DIR ${orbslam3_DIR}/Thirdparty/DBoW2)
file(GLOB DBOW2_SOURCES CONFIGURE_DEPENDS ${DBOW2_DIR}/DBoW2/*.cpp ${DBOW2_DIR}/DUtils/*.cpp)
add_library(dbow2 STATIC ${DBOW2_SOURCES})
target_include_directories(dbow2 PUBLIC ${DBOW2_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/compat)
target_link_libraries(dbow2 PUBLIC ${OpenCV_LIBS})
target_compile_options(dbow2 PRIVATE -w)

set(ORB_SOURCES Tracking.cc LocalMapping.cc ORBextractor.cc ORBmatcher.cc Frame.cc KeyFrame.cc MapPoint.cc
    Map.cc Atlas.cc KeyFrameDatabase.cc Optimizer.cc OptimizableTypes.cpp Converter.cc TwoViewReconstruction.cc ImuTypes.cc
    G2oTypes.cc MLPnPsolver.cpp GeometricTools.cc
    CameraModels/Pinhole.cpp CameraModels/KannalaBrandt8.cpp)
list(TRANSFORM ORB_SOURCES PREPEND ${orbslam3_DIR}/src/)
add_library(orbslam3 STATIC ${ORB_SOURCES} viobench_slam.cpp)
target_include_directories(orbslam3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${orbslam3_DIR} ${orbslam3_DIR}/include ${orbslam3_DIR}/include/CameraModels ${orbslam3_DIR}/Thirdparty/Sophus)
target_link_libraries(orbslam3 PUBLIC g2o dbow2 ${OpenCV_LIBS})
target_compile_options(orbslam3 PRIVATE -w)

# The reactor has no preopened filesystem: store the original trained vocabulary in linear memory.
if(CMAKE_SYSTEM_NAME STREQUAL "WASI" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(VOC_PLATFORM ${CMAKE_SYSTEM_NAME})
else()
    set(VOC_PLATFORM other)
endif()
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/vocabulary.S ${CMAKE_BINARY_DIR}/vocabulary.bin
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_vocabulary.py
            ${orbslam3_DIR}/Vocabulary/ORBvoc.txt.tar.gz ${CMAKE_BINARY_DIR}/vocabulary.bin ${CMAKE_BINARY_DIR}/vocabulary.S ${VOC_PLATFORM}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_vocabulary.py ${orbslam3_DIR}/Vocabulary/ORBvoc.txt.tar.gz
    VERBATIM)
viobench_solution(solution solution.cpp ${CMAKE_BINARY_DIR}/vocabulary.S)
target_link_libraries(solution PRIVATE orbslam3)
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    target_link_options(solution PRIVATE -fwasm-exceptions -Wl,--initial-memory=134217728 -Wl,-z,stack-size=16777216 -Wl,--stack-first)
    target_link_libraries(solution PRIVATE unwind)
endif()
