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

# find_package(FMath REQUIRED)
# find_package(glm CONFIG REQUIRED )

set(CMAKE_CXX_STANDARD 17)

if (USE_CUDA)
# SET( CUDA_PROPAGATE_HOST_FLAGS ON)
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
    FullExperimentRunner.cpp
    FullExperimentRunner.h

    FullExperimentRunnerOptimalPerturb.cpp
    FullExperimentRunnerOptimalPerturb.h
    # FullExperimentRunnerOptimalPerturbOptimized.cpp
    # FullExperimentRunnerOptimalPerturbOptimized.h

    # FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    # FullExperimentRunnerOptimalPerturbOptimized_GPU.h

    Integrate.h
    MathConsts.h
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
    Sample.h
    Sample.cpp
    Twisty_Cuda_Helpers.cu
    Twisty_Cuda_Helpers.h
)

if (USE_CUDA)

target_sources(Twisty
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/FullExperimentRunnerOptimalPerturbOptimized_GPU.h
)
target_compile_definitions(Twisty
    PUBLIC USE_CUDA
)

# set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES "35;50;72")

endif (USE_CUDA)

# set_target_properties(Twisty PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(Twisty PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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
add_executable(FullExperiment
    FullExperiment.cpp
)
else()
add_executable(FullExperiment
    FullExperiment.cpp
)
endif (WIN32)

target_compile_features(FullExperiment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FullExperiment
    PUBLIC Twisty
    PUBLIC glm
    PUBLIC libconfig++
    PUBLIC shlwapi
)

if (WIN32)
else()
target_compile_options(FullExperiment
    PUBLIC -lstdc++fs
    PUBLIC -std=c++17
)
target_link_options(FullExperiment
    PUBLIC -pthread
    PUBLIC -lstdc++fs
)
endif (WIN32)


# if (WIN32)
# add_executable(RingBenchmark
#     RingBenchmark.cpp
# )
# else()
# add_executable(RingBenchmark
#     RingBenchmark.cpp
# )
# endif (WIN32)

# target_compile_features(RingBenchmark
#     PUBLIC cxx_static_assert
#     PUBLIC cxx_std_17
# )

# target_link_libraries(RingBenchmark
#     PUBLIC Twisty
#     PUBLIC glm
#     PUBLIC libconfig++
#     PUBLIC shlwapi
# )

# if (WIN32)
# else()
# target_compile_options(RingBenchmark
#     PUBLIC -lstdc++fs
#     PUBLIC -std=c++17
# )
# target_link_options(RingBenchmark
#     PUBLIC -pthread
#     PUBLIC -lstdc++fs
# )
# endif (WIN32)


# Combined initial curve full experiment
# if (WIN32)
# add_executable(FullExperiment_CombinedInitialCurves
#     FullExperiment_CombinedInitialCurves.cpp
# )
# else()
# add_executable(FullExperiment_CombinedInitialCurves
#     FullExperiment_CombinedInitialCurves.cpp
# )
# endif (WIN32)

# target_compile_features(FullExperiment_CombinedInitialCurves
#     PUBLIC cxx_static_assert
#     PUBLIC cxx_std_17
# )

# target_link_libraries(FullExperiment_CombinedInitialCurves
#     PUBLIC Twisty
#     PUBLIC glm
# )

# if (WIN32)
# else()
# target_compile_options(FullExperiment_CombinedInitialCurves
#     PUBLIC -lstdc++fs
#     PUBLIC -std=c++17
# )
# target_link_options(FullExperiment_CombinedInitialCurves
#     PUBLIC -pthread
#     PUBLIC -lstdc++fs
# )
# endif (WIN32)



#  Freeze frame experiment

# add_executable(FreezeFrameExperiment
#     FreezeFrameExperiment.cpp
# )

# target_link_libraries(FreezeFrameExperiment
#     PUBLIC Twisty
#     PUBLIC glm
# )


# add_executable(FreezeFrameExperiment2
#     FreezeFrameExperiment2.cpp
# )

# target_link_libraries(FreezeFrameExperiment2
#     PUBLIC Twisty
#     PUBLIC glm
# )

# add_executable(NoisyCircleExperiment
#     NoisyCircleExperiment.cpp
# )

# target_link_libraries(NoisyCircleExperiment
#     PUBLIC Twisty
#     PUBLIC glm
# )

# add_executable(NoisyCircleExperimentHalf
#     NoisyCircleExperimentHalf.cpp
# )

# target_link_libraries(NoisyCircleExperimentHalf
#     PUBLIC Twisty
#     PUBLIC glm
# )


# add_executable(DifferentNormalSThetaExperiment
#     DifferentNormalSThetaExperiment.cpp
# )

# target_link_libraries(DifferentNormalSThetaExperiment
#     PUBLIC Twisty
#     PUBLIC glm
# )


# add_executable(Benchmark_5_1
#     Benchmark_5_1.cpp
# )

# target_link_libraries(Benchmark_5_1
#     PUBLIC Twisty
#     PUBLIC glm
# )

if (WIN32)
# add_subdirectory(Experiments)
add_subdirectory(Tools)
add_subdirectory(Viewer)
add_subdirectory(unitweights)
# add_subdirectory(Utils)
add_subdirectory(RelatedWork/Fred_2017/)
endif (WIN32)
