cmake_minimum_required(VERSION 3.20)

project(Twisty CXX)

include(FetchContent)

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" TRUE)
if (USE_CUDA)
enable_language(CUDA)

set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif (USE_CUDA)

# Needed for multiprecision float library
find_package(Boost)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(FMath
    GIT_REPOSITORY "https://github.com/BrennenTaylor/FMath"
)
FetchContent_MakeAvailable(FMath)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(Twisty
    BezierCurve.h
    BezierCurve.cpp
    Bootstrapper.h
    Bootstrapper.cpp
    CombinedWeightUtils.h
    # CombinedWeightUtils.cpp
    Curve.h
    Curve.cpp
    CurvePerturbUtils.h
    # CurvePerturbUtils.cpp
    CurveUtils.h
    CurveUtils.cpp
    Derivative.h
    ExperimentRunner.cpp
    ExperimentRunner.h

    FullExperimentRunnerOptimalPerturb.cpp
    FullExperimentRunnerOptimalPerturb.h

    FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    FullExperimentRunnerOptimalPerturbOptimized_GPU.h

    Integrate.h
    MathConsts.h
    PathWeighters.h
    # PathWeighters.cpp
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
)

if (USE_CUDA)
# set_source_files_properties(CombinedWeightUtils.cpp PROPERTIES LANGUAGE CUDA)
# set_source_files_properties(CurvePerturbUtils.cpp PROPERTIES LANGUAGE CUDA)
# set_source_files_properties(PathWeighters.cpp PROPERTIES LANGUAGE CUDA)
else()
endif(USE_CUDA)

if (USE_CUDA)
set (cuda_flags -Xcompiler=-Wall)
# set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES 61 70 72)
set_target_properties(Twisty PROPERTIES CUDA_STANDARD 17)
# set_target_properties(Twisty PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(Twisty PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(Twisty PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode=arch=compute_61,code=[compute_61,lto_61] -gencode=arch=compute_70,code=[compute_70,lto_70] -gencode=arch=compute_72,code=[compute_72,lto_72] -Xptxas -v>)
target_link_options(Twisty PRIVATE $<DEVICE_LINK:-gencode=arch=compute_61,code=sm_61 -gencode=arch=70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -dlto>)

target_compile_definitions(Twisty
    PUBLIC USE_CUDA
)
endif (USE_CUDA)

target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
)
if (USE_CUDA)
target_include_directories(Twisty
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
endif(USE_CUDA)

target_compile_features(Twisty
    PUBLIC cxx_static_assert
)

target_compile_options(Twisty
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/openmp>>
)

if (WIN32)
else()
target_compile_options(Twisty
    PUBLIC -lstdc++fs
)
target_link_options(Twisty
    PUBLIC -pthread
    PUBLIC -lstdc++fs
)
endif (WIN32)

target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC shlwapi
    PUBLIC nlohmann_json::nlohmann_json
)

if (WIN32)
add_subdirectory(Experiments)
# add_subdirectory(Tools)
# add_subdirectory(Utils)
add_subdirectory(Viewer)
add_subdirectory(unitweights)
add_subdirectory(RelatedWork/Fred_2017/)
endif (WIN32)
