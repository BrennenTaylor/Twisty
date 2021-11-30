cmake_minimum_required(VERSION 3.12)

project(Twisty CXX)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" TRUE)
if (USE_CUDA)
SET(CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
endif (USE_CUDA)

# Needed for multiprecision float library
find_package(Boost)
# Needed for config files
find_package(RapidJSON CONFIG REQUIRED)

# Needed for math operations
add_subdirectory(FMath)
# Needed for config files
add_subdirectory(libconfig)

IF (WIN32)
add_subdirectory(tinyexr)
ENDIF()

set(CMAKE_CXX_STANDARD 17)

if (USE_CUDA)
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif (USE_CUDA)

add_library(Twisty
    BezierCurve.h
    BezierCurve.cpp
    Bootstrapper.h
    Bootstrapper.cpp
    Curve.h
    Curve.cpp
    CurvePerturbUtils.h
    CurvePerturbUtils.cu
    CurveUtils.h
    CurveUtils.cpp
    CurveGen.cu
    Derivative.h
    DeviceCurve.h
    ExperimentRunner.cpp
    ExperimentRunner.h

    FullExperimentRunnerOptimalPerturb.cpp
    FullExperimentRunnerOptimalPerturb.h

    Integrate.h
    MathConsts.h
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
)

if (USE_CUDA)
target_sources(Twisty
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FullExperimentRunnerOptimalPerturbOptimized_GPU.h
)
target_compile_definitions(Twisty
    PUBLIC USE_CUDA
)
endif (USE_CUDA)

target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
    PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
)

target_compile_features(Twisty
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_compile_options(Twisty
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/openmp>>
)

if (WIN32)
else()
target_compile_options(Twisty
    PUBLIC -lstdc++fs
    PUBLIC -std=c++17
)
target_link_options(Twisty
    PUBLIC -pthread
    PUBLIC -lstdc++fs
)
endif (WIN32)

target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC libconfig++
    PUBLIC shlwapi
)

if (WIN32)
add_subdirectory(Experiments)
add_subdirectory(Tools)
add_subdirectory(Viewer)
add_subdirectory(unitweights)
add_subdirectory(RelatedWork/Fred_2017/)
endif (WIN32)
