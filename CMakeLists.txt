cmake_minimum_required(VERSION 3.12)

project(Twisty CXX)


find_package(Boost)
if (WIN32)
find_package(CUDA REQUIRED)
endif (WIN32)

find_package(FMath REQUIRED)
find_package(fmt REQUIRED)
find_package(glm CONFIG REQUIRED )
find_package(RapidJSON CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
SET( CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
set(CUDA_cudadevrt_LIBRARY PUBLIC ${CUDA_cudadevrt_LIBRARY})
endif (WIN32)

if (WIN32)
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
    # ExperimentRunnerCpu.cpp
    # ExperimentRunnerCpu.h
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
    # ExperimentRunnerCpu.cpp
    # ExperimentRunnerCpu.h
    # ExperimentRunnerGpu.cu
    # ExperimentRunnerGpu.h
    FullExperimentRunner.cpp
    FullExperimentRunner.h
    Geometry.h
    Geometry.cpp
    GeometryBootstrapper.h
    GeometryBootstrapper.cpp
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
)
endif (WIN32)

target_include_directories(Twisty
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
    PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
)

target_compile_features(Twisty
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if (WIN32)
target_compile_options(Twisty
    PRIVATE /openmp
    PUBLIC /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
)
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
    PUBLIC fmt::fmt
    PUBLIC fmt::fmt-header-only
    PUBLIC glm
)



# if (WIN32)
# cuda_add_executable(PathGeneratorCpu
#     Main_CPU.cpp
# )
# else()
# add_executable(PathGeneratorCpu
#     Main_CPU.cpp
# )
# endif (WIN32)

# target_compile_features(PathGeneratorCpu
#     PUBLIC cxx_static_assert
#     PUBLIC cxx_std_17
# )

# target_link_libraries(PathGeneratorCpu
#     Twisty
# )

if (WIN32)
cuda_add_executable(FullExperiment
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

add_executable(FreezeFrameExperiment
    FreezeFrameExperiment.cpp
)

target_compile_features(FreezeFrameExperiment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FreezeFrameExperiment
    PUBLIC Twisty
    PUBLIC glm
)

if (WIN32)
else()
target_compile_options(FreezeFrameExperiment
    PUBLIC -lstdc++fs
    PUBLIC -std=c++17
)
target_link_options(FreezeFrameExperiment
    PUBLIC -pthread
    PUBLIC -lstdc++fs
)
endif (WIN32)

# add_subdirectory(Experiments)
if (WIN32)
add_subdirectory(Tests)
add_subdirectory(Tools)
add_subdirectory(Viewer)
add_subdirectory(ThirdParty)
add_subdirectory(Utils)
endif (WIN32)
