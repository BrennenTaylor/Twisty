cmake_minimum_required(VERSION 3.20)

project(Neural LANGUAGES CXX)

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

add_subdirectory(dependencies)

find_package(OpenVDB CONFIG REQUIRED)

add_executable(raymarch_proof
    raymarch_proof.cpp
)

target_link_libraries(raymarch_proof
    PUBLIC FMath::FMath
    PUBLIC tinyexr_include
)

set_property(TARGET raymarch_proof PROPERTY CXX_STANDARD 20)

add_executable(generate_dataset_multiple_light_positions
    generate_dataset_multiple_light_positions.cpp
)

target_link_libraries(generate_dataset_multiple_light_positions
    PUBLIC FMath::FMath
    PUBLIC tinyexr_include
)

set_property(TARGET generate_dataset_multiple_light_positions PROPERTY CXX_STANDARD 20)

add_executable(raymarch_vdb
    raymarch_vdb.cpp
)

target_link_libraries(raymarch_vdb
    PUBLIC FMath::FMath
    PUBLIC tinyexr_include
    PRIVATE OpenVDB::openvdb
)

set_property(TARGET raymarch_vdb PROPERTY CXX_STANDARD 20)

add_executable(test_sph
    test_sph.cpp
)

target_link_libraries(test_sph
    PUBLIC FMath::FMath
)

set_property(TARGET test_sph PROPERTY CXX_STANDARD 20)