add_executable(FullExperiment
    FullExperiment.cpp
)

target_link_libraries(FullExperiment
    PUBLIC Twisty
)

add_executable(StressTestCombinedWeights
    StressTestCombinedWeights.cpp
)

target_link_libraries(StressTestCombinedWeights
    PUBLIC Twisty
)

add_executable(NoisyCircleAngleIntegration
    NoisyCircleAngleIntegration.cpp
)

target_link_libraries(NoisyCircleAngleIntegration
    PUBLIC Twisty
)

# Refactor below
add_executable(NoisyCircleExperimentHalf
    NoisyCircleExperimentHalf.cpp
)

target_compile_features(NoisyCircleExperimentHalf
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(NoisyCircleExperimentHalf
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(BeamSpreadExperiment
    BeamSpreadExperiment.cpp
)

target_compile_features(BeamSpreadExperiment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(BeamSpreadExperiment
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(NoisyCircleExperiment
    NoisyCircleExperiment.cpp
)

target_compile_features(NoisyCircleExperiment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(NoisyCircleExperiment
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(RingBenchmark
    RingBenchmark.cpp
)

target_compile_features(RingBenchmark
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(RingBenchmark
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(FiveSegmentExploreDoF
    FiveSegmentExploreDoF.cpp
)

target_compile_features(FiveSegmentExploreDoF
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FiveSegmentExploreDoF
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

target_include_directories(FiveSegmentExploreDoF
    PUBLIC ../stb
)

add_executable(SixSegmentExploreDoF
    SixSegmentExploreDoF.cpp
)

target_compile_features(SixSegmentExploreDoF
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(SixSegmentExploreDoF
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

target_include_directories(SixSegmentExploreDoF
    PUBLIC ../stb
)

add_executable(FiveSegmentHeatmap
    FiveSegmentHeatmap.cpp
)

target_compile_features(FiveSegmentHeatmap
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FiveSegmentHeatmap
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

target_include_directories(FiveSegmentHeatmap
    PUBLIC ../stb
)

add_executable(FiveSegmentAnglePathIntegral
    FiveSegmentAnglePathIntegral.cpp
)

target_compile_features(FiveSegmentAnglePathIntegral
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(FiveSegmentAnglePathIntegral
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(FiveSegmentAnglePathIntegral
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(SixSegmentAnglePathIntegral
    SixSegmentAnglePathIntegral.cpp
)

target_compile_features(SixSegmentAnglePathIntegral
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(SixSegmentAnglePathIntegral
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(FiveSegmentUniformPathGenerationSolver
    FiveSegmentUniformPathGenerationSolver.cpp
)

target_compile_features(FiveSegmentUniformPathGenerationSolver
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(FiveSegmentUniformPathGenerationSolver
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(FiveSegmentUniformPathGenerationSolver
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)