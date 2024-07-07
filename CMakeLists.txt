cmake_minimum_required(VERSION 3.20)

project(Twisty LANGUAGES CXX)

set(Twisty_VERSION 1.0.0)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (NOT LINUX)
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

endif()

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" FALSE)

set(OPENVDB_CORE_STATIC OFF)
add_subdirectory(openvdb)

add_subdirectory(dependencies)
add_subdirectory(Twisty)

add_subdirectory(Tools)
add_subdirectory(Experiments)

if(WIN32)
    add_subdirectory(Tests)
    add_subdirectory(Utils)
    add_subdirectory(Viewer)
    add_subdirectory(unitweights)
    add_subdirectory(RelatedWork/Fred_2017/)
endif(WIN32)
