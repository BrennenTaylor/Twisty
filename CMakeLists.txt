cmake_minimum_required(VERSION 3.20)

project(Twisty VERSION 1.0.0 LANGUAGES CXX)

include(cmake/TwistyHelpers.cmake)

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(WIN32)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)UseMultiToolTask=")
        list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    endif()

    if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)EnforceProcessCountAcrossBuilds=")
        list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
    endif()
endif()

add_subdirectory(dependencies)
add_subdirectory(Twisty)
add_subdirectory(Tools)
add_subdirectory(Experiments)

enable_testing()
add_subdirectory(Tests)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "52;70;72")
endif()
