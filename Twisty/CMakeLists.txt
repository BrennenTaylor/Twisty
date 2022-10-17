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

target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC shlwapi
    PUBLIC nlohmann_json::nlohmann_json
)

target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${Boost_INCLUDE_DIR}
    PUBLIC $<$<BOOL:${USE_CUDA}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

target_compile_options(Twisty
    PUBLIC $<$<NOT:$<BOOL:${WIN32}>>:"-lstdc++fs">
)

# We dont want to generate this on windows atm. Linker is only non-windows platforms
if(NOT WIN32)
    target_link_options(Twisty
        PUBLIC "-pthread -lstdc++fs" >
    )
endif()

if(USE_CUDA)
    set_target_properties(Twisty PROPERTIES CUDA_ARCHITECTURES "52;70;72")
    target_compile_options(Twisty
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xptxas -v>
    )
    target_compile_definitions(Twisty PUBLIC USE_CUDA)
endif(USE_CUDA)