cmake_minimum_required(VERSION 3.12)

project(Twisty CXX)
if (WIN32)
# find_package(CUDA REQUIRED)
enable_language(CUDA)
endif (WIN32)

find_package(Boost)
find_package(RapidJSON CONFIG REQUIRED)

add_subdirectory(FMath)

IF (WIN32)
add_subdirectory(tinyexr)
ENDIF()

# find_package(FMath REQUIRED)
# find_package(glm CONFIG REQUIRED )

# set(CMAKE_CXX_STANDARD 17)

# if (WIN32)
# SET( CUDA_PROPAGATE_HOST_FLAGS ON)
# SET(CUDA_SEPARABLE_COMPILATION ON)
# SET(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
# set(CUDA_cudadevrt_LIBRARY PUBLIC ${CUDA_cudadevrt_LIBRARY})
# endif (WIN32)

if (WIN32)
add_library(Twisty
    BezierCurve.h
    BezierCurve.cpp
    Bootstrapper.h
    Bootstrapper.cpp
    Curve.h
    Curve.cpp
    CurvePurturber.h
    CurvePurturber.cpp
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
    FullExperimentRunner2.cpp
    FullExperimentRunner2.h

    FullExperimentRunnerOptimalPerturb.cpp
    FullExperimentRunnerOptimalPerturb.h
    FullExperimentRunnerOptimalPerturbOptimized.cpp
    FullExperimentRunnerOptimalPerturbOptimized.h

    FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    FullExperimentRunnerOptimalPerturbOptimized_GPU.h

    FullExperimentRunnerOldMethodBridge.cpp
    FullExperimentRunnerOldMethodBridge.h
    Geometry.h
    Geometry.cpp
    GeometryBootstrapper.h
    GeometryBootstrapper.cpp

    Integrate.h
    MathConsts.h
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
    Range.h
    Sample.h
    Sample.cpp
    SpecifiedCurveParamBootstrapper.h
    SpecifiedCurveParamBootstrapper.cpp
    StartEndBootstrapper.h
    StartEndBootstrapper.cpp
    TestBootstrappers.h
    TestBootstrappers.cpp
    Twisty_Cuda_Helpers.cu
    Twisty_Cuda_Helpers.h
)
else()
add_library(Twisty
    BezierCurve.h
    BezierCurve.cpp
    Bootstrapper.h
    Bootstrapper.cpp
    Curve.h
    Curve.cpp
    CurvePurturber.h
    CurvePurturber.cpp
    CurvePerturbUtils.h
    CurvePerturbUtils_linux.cpp
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
    FullExperimentRunnerOptimalPerturbOptimized.cpp
    FullExperimentRunnerOptimalPerturbOptimized.h
    Geometry.h
    Geometry.cpp
    GeometryBootstrapper.h
    GeometryBootstrapper.cpp
    Integrate.h
    MathConsts.h
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
    Range.h
    Sample.h
    Sample.cpp
    SpecifiedCurveParamBootstrapper.h
    SpecifiedCurveParamBootstrapper.cpp
    StartEndBootstrapper.h
    StartEndBootstrapper.cpp
    TestBootstrappers.h
    TestBootstrappers.cpp
    Twisty_Cuda_Helpers.cu
    Twisty_Cuda_Helpers.h
)
endif (WIN32)

set_target_properties(Twisty PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Twisty PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(Twisty PROPERTIES CUDA_STANDARD 20)
set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES 0)

target_include_directories(Twisty
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
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


# Combined initial curve full experiment
if (WIN32)
add_executable(FullExperiment_CombinedInitialCurves
    FullExperiment_CombinedInitialCurves.cpp
)
else()
add_executable(FullExperiment_CombinedInitialCurves
    FullExperiment_CombinedInitialCurves.cpp
)
endif (WIN32)

target_compile_features(FullExperiment_CombinedInitialCurves
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FullExperiment_CombinedInitialCurves
    PUBLIC Twisty
    PUBLIC glm
)

if (WIN32)
else()
target_compile_options(FullExperiment_CombinedInitialCurves
    PUBLIC -lstdc++fs
    PUBLIC -std=c++17
)
target_link_options(FullExperiment_CombinedInitialCurves
    PUBLIC -pthread
    PUBLIC -lstdc++fs
)
endif (WIN32)



#  Freeze frame experiment

add_executable(FreezeFrameExperiment
    FreezeFrameExperiment.cpp
)

target_link_libraries(FreezeFrameExperiment
    PUBLIC Twisty
    PUBLIC glm
)


add_executable(FreezeFrameExperiment2
    FreezeFrameExperiment2.cpp
)

target_link_libraries(FreezeFrameExperiment2
    PUBLIC Twisty
    PUBLIC glm
)

add_executable(NoisyCircleExperiment
    NoisyCircleExperiment.cpp
)

target_link_libraries(NoisyCircleExperiment
    PUBLIC Twisty
    PUBLIC glm
)

add_executable(NoisyCircleExperimentHalf
    NoisyCircleExperimentHalf.cpp
)

target_link_libraries(NoisyCircleExperimentHalf
    PUBLIC Twisty
    PUBLIC glm
)


add_executable(DifferentNormalSThetaExperiment
    DifferentNormalSThetaExperiment.cpp
)

target_link_libraries(DifferentNormalSThetaExperiment
    PUBLIC Twisty
    PUBLIC glm
)


add_executable(Benchmark_5_1
    Benchmark_5_1.cpp
)

target_link_libraries(Benchmark_5_1
    PUBLIC Twisty
    PUBLIC glm
)

if (WIN32)
# add_subdirectory(Experiments)
add_subdirectory(Tests)
add_subdirectory(Tools)
add_subdirectory(Viewer)
add_subdirectory(Utils)
endif (WIN32)
