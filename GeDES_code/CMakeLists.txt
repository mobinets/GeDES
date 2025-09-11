cmake_minimum_required(VERSION 3.0.0)
project(GeDES LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug 模式标志
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -std=c++17")

# Release 模式标志
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -std=c++17")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -std=c++17")

include_directories(
    "/path/to/source/document/GeDES_code/code/c++/include/"
    "/usr/local/cuda/include"
    "/usr/include"
    "/usr/local/include"
)

find_package(CUDA REQUIRED)
find_library(CUDA_LIB cuda PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

set(SOURCE_DIR /path/to/source/document/GeDES_code/code/c++/src/)
file(GLOB_RECURSE SOURCES ${SOURCE_DIR}/*.cc)
file(GLOB_RECURSE CUDA_SOURCES ${SOURCE_DIR}/*.cu)

add_executable(GeDES
    main.cc
    ${SOURCES}
    ${CUDA_SOURCES}

)

set_target_properties(GeDES PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

set_property(TARGET GeDES PROPERTY CXX_STANDARD 17)

# Address sanitizer
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)
