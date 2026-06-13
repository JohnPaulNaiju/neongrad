cmake_minimum_required(VERSION 3.20)

project(deeplearning-engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type" FORCE)
endif()

add_library(lodepng STATIC third_party/lodepng/lodepng.cpp)
target_include_directories(lodepng PUBLIC third_party/lodepng)

add_library(nn_core STATIC
        src/core/tensor.cpp
)
target_include_directories(nn_core PUBLIC src)
target_link_libraries(nn_core PUBLIC lodepng)

add_executable(nn_train src/main_train.cpp)
target_link_libraries(nn_train PRIVATE nn_core)

enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE nn_core)
add_test(NAME TensorTest COMMAND test_tensor)