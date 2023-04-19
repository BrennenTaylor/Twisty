find_package(OpenMP)

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

    # FullExperimentRunnerOptimalPerturbOptimized_GPU.cu
    # FullExperimentRunnerOptimalPerturbOptimized_GPU.h
    Integrate.h
    MathConsts.h
    PathWeighters.h

    # CombinedWeightUtils.cpp
    # CurvePerturbUtils.cpp
    PathWeighters.cpp
    PathWeightUtils.h
    PathWeightUtils.cpp
    PerturbUtils.h
    PerturbUtils.cpp
)

target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC OpenMP::OpenMP_CXX
)
if(WIN32)
target_link_libraries(Twisty
    PUBLIC shlwapi
)
endif()

target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
    PUBLIC $<$<BOOL:${USE_CUDA}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

if(USE_CUDA)
    set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES "52;70;72")
    target_compile_options(Twisty
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xptxas -v>
    )
    target_compile_definitions(Twisty PUBLIC USE_CUDA)
endif(USE_CUDA)