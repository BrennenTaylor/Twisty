# tinyexr: fetched headers only; a single TINYEXR_IMPLEMENTATION TU
# (tinyexr_impl.cpp) compiles the implementation into ExperimentBase, and the
# bundled miniz is built as a small static lib to satisfy its deflate needs.
FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY "https://github.com/syoyo/tinyexr"
    GIT_TAG "release"
)

FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()

add_library(tinyexr_interface INTERFACE)
target_include_directories(tinyexr_interface INTERFACE ${tinyexr_SOURCE_DIR}
${tinyexr_SOURCE_DIR}/deps/miniz
)

enable_language(C)
add_library(miniz STATIC ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(miniz PUBLIC ${tinyexr_SOURCE_DIR}/deps/miniz)
set_property(TARGET miniz PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tinyexr_interface INTERFACE miniz)

add_library(ExperimentBase
    ExperimentBase.h
    ExperimentBase.cpp
    ExperimentUtils.h
    ExperimentUtils.cpp
    tinyexr_impl.cpp
)

target_include_directories(ExperimentBase INTERFACE .)

target_compile_features(ExperimentBase
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(ExperimentBase
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PUBLIC tinyexr_interface
    PRIVATE nlohmann_json::nlohmann_json
)

if(LINUX)
    target_link_libraries(ExperimentBase
        PUBLIC OpenVDB::openvdb
    )
endif()

# --- Experiments ---
twisty_add_experiment(FullExperiment                        FullExperiment.cpp)
twisty_add_experiment(StressTestCombinedWeights             StressTestCombinedWeights.cpp)
twisty_add_experiment(NoisyCircleAngleIntegration           NoisyCircleAngleIntegration.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCircleExperimentHalf             NoisyCircleExperimentHalf.cpp)
twisty_add_experiment(BeamSpreadExperiment                  BeamSpreadExperiment.cpp)
twisty_add_experiment(NoisyCircleExperiment                 NoisyCircleExperiment.cpp)
twisty_add_experiment(RingBenchmark                         RingBenchmark.cpp)
twisty_add_experiment(FiveSegmentExploreDoF                 FiveSegmentExploreDoF.cpp STB)
twisty_add_experiment(SixSegmentExploreDoF                  SixSegmentExploreDoF.cpp STB)
twisty_add_experiment(FiveSegmentHeatmap                    FiveSegmentHeatmap.cpp STB)
twisty_add_experiment(FiveSegmentAnglePathIntegral          FiveSegmentAnglePathIntegral.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentAnglePathIntegral           SixSegmentAnglePathIntegral.cpp EXPERIMENT_BASE)
twisty_add_experiment(FiveSegmentUniformPathGenerationSolver FiveSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentUniformPathGenerationSolver  SixSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(MSegmentUniformPathGenerationSolver    MSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(FiveSegmentAngleSpaceMC               FiveSegmentAngleSpaceMC.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentAngleSpaceMC                SixSegmentAngleSpaceMC.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment     NoisyCirclePathGenerationMSegment.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment_Pinhole NoisyCirclePathGenerationMSegment_Pinhole.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment_RaycastVolume NoisyCirclePathGenerationMSegment_RaycastVolume.cpp EXPERIMENT_BASE)
twisty_add_experiment(Simple_FiveSegment_Analytical         Simple_FiveSegment_Analytical.cpp EXPERIMENT_BASE)
twisty_add_experiment(Simple_SixSegment_Analytical          Simple_SixSegment_Analytical.cpp EXPERIMENT_BASE)
twisty_add_experiment(CalculateNormalizer                   CalculateNormalizer.cpp EXPERIMENT_BASE)
