cmake_minimum_required(VERSION 3.12)

project(Twisty CXX)

find_package(BOOST REQUIRED)
find_package(CUDA REQUIRED)
find_package(FMath REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)

SET( CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

cuda_add_library(Twisty
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
    ExperimentRunnerCpu.cpp
    ExperimentRunnerCpu.h
    # ExperimentRunnerGpu.cu
    # ExperimentRunnerGpu.h
    FullExperimentRunner.cpp
    FullExperimentRunner.h
    Geometry.h
    Geometry.cpp
    GeometryBootstrapper.h
    GeometryBootstrapper.cpp
    # GpuFullExperimentRunnerOptimized.cu
    # GpuFullExperimentRunnerOptimized.h
    GpuFullExperimentRunnerGeneral.cu
    GpuFullExperimentRunnerGeneral.h
    Integrate.h
    MathConsts.h
    PathWeightUtils.h
    PathWeightUtils.cpp
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
    TwistyYamlUtils.cpp
    TwistyYamlUtils.h
)

target_include_directories(Twisty
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
    PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
)

target_compile_features(Twisty
    PUBLIC cxx_static_assert
)

target_compile_options(Twisty
    PUBLIC /std:c++17
    PRIVATE /openmp
    PUBLIC /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
)

target_link_libraries(Twisty
    FMath::FMath
    yaml-cpp
    fmt::fmt
    fmt::fmt-header-only
)

cuda_add_executable(PathGeneratorCpu
    Main_CPU.cpp
)

target_compile_features(PathGeneratorCpu
    PUBLIC cxx_static_assert
)

target_compile_options(PathGeneratorCpu
    PUBLIC /std:c++17)


target_link_libraries(PathGeneratorCpu
    Twisty
)

cuda_add_executable(FullExperiment
    FullExperiment.cpp
)

target_compile_features(FullExperiment
    PUBLIC cxx_static_assert
)

target_compile_options(FullExperiment
    PUBLIC /std:c++17)


target_link_libraries(FullExperiment
    Twisty
)

cuda_add_executable(FreezeFrameExperiment
    FreezeFrameExperiment.cpp
)

target_compile_features(FreezeFrameExperiment
    PUBLIC cxx_static_assert
)

target_compile_options(FreezeFrameExperiment
    PUBLIC /std:c++17)


target_link_libraries(FreezeFrameExperiment
    Twisty
)

# add_subdirectory(Experiments)
add_subdirectory(Tests)
add_subdirectory(Tools)
add_subdirectory(Viewer)
add_subdirectory(ThirdParty)
add_subdirectory(Utils)