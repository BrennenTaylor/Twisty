cmake_minimum_required(VERSION 3.20)

project(Twisty LANGUAGES CXX)

set(Twisty_VERSION 1.0.0)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)UseMultiToolTask=")
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
endif()

if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)EnforceProcessCountAcrossBuilds=")
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" FALSE)

# if(USE_CUDA)
#     enable_language(CUDA)
#     set(CUDA_VERBOSE_BUILD ON)
#     set(CMAKE_CUDA_STANDARD 17)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif(USE_CUDA)

add_subdirectory(dependencies)

# add_subdirectory(Neural)
add_subdirectory(Twisty)

if(WIN32)
    add_subdirectory(Experiments)
    add_subdirectory(Tests)
    add_subdirectory(Tools)
    add_subdirectory(Utils)
    add_subdirectory(Viewer)
    add_subdirectory(unitweights)
    add_subdirectory(RelatedWork/Fred_2017/)
endif(WIN32)
