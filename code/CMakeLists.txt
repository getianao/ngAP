cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
enable_language(CXX CUDA)

project(GPUNFA2019)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -lineinfo --expt-extended-lambda -O3 -Wno-deprecated-gpu-targets -arch=sm_86 --keep")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -lineinfo --expt-extended-lambda -O3 -Wno-deprecated-gpu-targets -arch=sm_70 --keep")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -lineinfo --expt-extended-lambda -g -G -Wno-deprecated-gpu-targets -arch=sm_86")
# -arch=sm_86

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_definitions(-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)

include_directories(.)
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/commons/*")
add_library(gpunfacommons ${SOURCES})
target_link_libraries(gpunfacommons tbb)
target_include_directories(gpunfacommons PRIVATE include/commons include/pugixml )


file(GLOB_RECURSE SOURCES1 RELATIVE ${CMAKE_SOURCE_DIR} "src/gpunfautils/*")
cuda_add_library(gpunfautils ${SOURCES1})
target_include_directories(gpunfautils PRIVATE include/commons include/gpunfautils)
set_property(TARGET gpunfautils PROPERTY CUDA_STANDARD 17)


# CUDA Projects
# obat
SET(PROJ "obat")
file(GLOB_RECURSE SOURCES2 RELATIVE ${CMAKE_SOURCE_DIR} "src/${PROJ}/*")
cuda_add_executable(${PROJ} ${SOURCES2})
set_property(TARGET ${PROJ} PROPERTY CUDA_STANDARD 11)
target_link_libraries(${PROJ} gpunfacommons gpunfautils)
target_include_directories(${PROJ} PRIVATE include/commons;include/gpunfautils)

# # infant
# file(GLOB_RECURSE SOURCES_INFANT RELATIVE ${CMAKE_SOURCE_DIR} "src/infant/*")
# cuda_add_executable(infant ${SOURCES_INFANT})
# set_property(TARGET infant PROPERTY CUDA_STANDARD 11)
# target_link_libraries(infant gpunfacommons gpunfautils)

# ppopp12
file(GLOB_RECURSE SOURCES_PPOPP RELATIVE ${CMAKE_SOURCE_DIR} "src/ppopp12/*")
cuda_add_executable(ppopp12 ${SOURCES_PPOPP})
set_property(TARGET ppopp12 PROPERTY CUDA_STANDARD 11)
target_link_libraries(ppopp12 gpunfacommons gpunfautils)
target_include_directories(ppopp12 PRIVATE include/commons;include/gpunfautils)


if(DEFINED DATA_BUFFER_SIZE)
  message("DATA_BUFFER_SIZE is defined to ${DATA_BUFFER_SIZE}")
  add_compile_definitions(DATA_BUFFER_SIZE=${DATA_BUFFER_SIZE})
endif()
if(DEFINED DATA_BUFFER_SIZE_FRONTIER)
  message("DATA_BUFFER_SIZE_FRONTIER is defined to ${DATA_BUFFER_SIZE_FRONTIER}")
  add_compile_definitions(DATA_BUFFER_SIZE_FRONTIER=${DATA_BUFFER_SIZE_FRONTIER})
endif()
if(DEFINED RESULTS_SIZE)
  message("RESULTS_SIZE is defined to ${RESULTS_SIZE}")
  add_compile_definitions(RESULTS_SIZE=${RESULTS_SIZE})
endif()


# ngap
file(GLOB_RECURSE SOURCES_PPOPP RELATIVE ${CMAKE_SOURCE_DIR} "src/ngap/*")
cuda_add_executable(ngap ${SOURCES_PPOPP})
set_property(TARGET ngap PROPERTY CUDA_STANDARD 11)
target_link_libraries(ngap gpunfacommons gpunfautils)
target_include_directories(ngap PRIVATE include/commons;include/gpunfautils)

# asyncap
add_subdirectory(src/asyncap)
add_custom_target(asyncap ALL
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/src/asyncap all
    COMMENT "Running Makefile in asyncap"
)
add_dependencies(asyncap gpunfautils gpunfacommons)
set_property(
    TARGET asyncap
    APPEND
    PROPERTY ADDITIONAL_CLEAN_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asyncap/bin
    ${CMAKE_CURRENT_SOURCE_DIR}/build/bin/asyncap
)



