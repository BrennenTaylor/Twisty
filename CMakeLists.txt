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
option(USE_CUDA "Enable Cuda and GPU runner?" TRUE)

if(USE_CUDA)
    enable_language(CUDA)
    set(CUDA_VERBOSE_BUILD ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif(USE_CUDA)

add_subdirectory(dependencies)

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

if(USE_CUDA)
    set(cuda_flags -Xcompiler=-Wall)

    set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES "52;70;72")

    target_compile_options(Twisty PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xptxas -v>)

    target_compile_definitions(Twisty
        PUBLIC USE_CUDA
    )
endif(USE_CUDA)

target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
)

if(USE_CUDA)
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

if(WIN32)
else()
    target_compile_options(Twisty
        PUBLIC -lstdc++fs
    )
    target_link_options(Twisty
        PUBLIC -pthread
        PUBLIC -lstdc++fs
    )
endif(WIN32)

target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC shlwapi
    PUBLIC nlohmann_json::nlohmann_json
)

if(WIN32)
    add_subdirectory(Experiments)

    add_subdirectory(Tools)
    add_subdirectory(Utils)
    add_subdirectory(Viewer)
    add_subdirectory(unitweights)
    add_subdirectory(RelatedWork/Fred_2017/)
endif(WIN32)
