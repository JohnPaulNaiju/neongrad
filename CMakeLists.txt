cmake_minimum_required(VERSION 3.20)

project(deeplearning-engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type" FORCE)
endif()

set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
find_package(OpenMP REQUIRED)

add_library(lodepng STATIC third_party/lodepng/lodepng.cpp)
target_include_directories(lodepng PUBLIC third_party/lodepng)

add_library(nn_core STATIC
        src/core/tensor.cpp
        src/core/tensor_math.cpp
        src/core/gemm_naive.cpp
        src/nn/layer.cpp
        src/nn/initializer.cpp
        src/nn/activations.cpp
)

set_source_files_properties(src/core/gemm_naive.cpp
        PROPERTIES COMPILE_FLAGS "-fno-vectorize -fno-slp-vectorize"
)

target_include_directories(nn_core PUBLIC src)
target_link_libraries(nn_core PUBLIC lodepng)

if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(nn_train src/main_train.cpp)
target_link_libraries(nn_train PRIVATE nn_core)

add_executable(benchmark_gemm benchmarks/benchmark_gemm.cpp)
target_link_libraries(benchmark_gemm PRIVATE nn_core)

option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE nn_core)
    add_test(NAME TensorTest COMMAND test_tensor)
endif()