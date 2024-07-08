# if(LINUX)
#     set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/lib/cmake/OpenVDB/")

#     # set(OPENVDB_DIR "/usr/local/lib/cmake/OpenVDB/")
# endif()
    # find_package(OpenVDB REQUIRED)

FetchContent_Declare(tinyexr
    GIT_REPOSITORY "https://github.com/syoyo/tinyexr"
    GIT_TAG "release"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(tinyexr)

# add_library(tinyexr_include INTERFACE)
# target_include_directories(tinyexr_include INTERFACE ${tinyexr_SOURCE_DIR}/)
# target_link_libraries(tinyexr_include INTERFACE miniz)

add_library(ExperimentBase
    ExperimentBase.h
    ExperimentBase.cpp
    ExperimentUtils.h
    ExperimentUtils.cpp
)

target_include_directories(ExperimentBase INTERFACE .
)


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
    PUBLIC tinyexr_include
    PRIVATE nlohmann_json::nlohmann_json
)

# if(LINUX)
    target_link_libraries(ExperimentBase
        PUBLIC openvdb
    )
# endif()

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
    PUBLIC ExperimentBase
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
    PUBLIC ExperimentBase
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
    PUBLIC ExperimentBase
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
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(SixSegmentUniformPathGenerationSolver
    SixSegmentUniformPathGenerationSolver.cpp
)

target_compile_features(SixSegmentUniformPathGenerationSolver
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(SixSegmentUniformPathGenerationSolver
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(SixSegmentUniformPathGenerationSolver
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(MSegmentUniformPathGenerationSolver
    MSegmentUniformPathGenerationSolver.cpp
)

target_compile_features(MSegmentUniformPathGenerationSolver
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(MSegmentUniformPathGenerationSolver
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(MSegmentUniformPathGenerationSolver
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(FiveSegmentAngleSpaceMC
    FiveSegmentAngleSpaceMC.cpp
)

target_compile_features(FiveSegmentAngleSpaceMC
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(FiveSegmentAngleSpaceMC
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(FiveSegmentAngleSpaceMC
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(SixSegmentAngleSpaceMC
    SixSegmentAngleSpaceMC.cpp
)

target_compile_features(SixSegmentAngleSpaceMC
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(SixSegmentAngleSpaceMC
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(SixSegmentAngleSpaceMC
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(NoisyCirclePathGenerationMSegment
    NoisyCirclePathGenerationMSegment.cpp
)

target_compile_features(NoisyCirclePathGenerationMSegment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(NoisyCirclePathGenerationMSegment
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(NoisyCirclePathGenerationMSegment
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(NoisyCirclePathGenerationMSegment_Pinhole
    NoisyCirclePathGenerationMSegment_Pinhole.cpp
)

target_compile_features(NoisyCirclePathGenerationMSegment_Pinhole
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(NoisyCirclePathGenerationMSegment_Pinhole
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(NoisyCirclePathGenerationMSegment_Pinhole
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

# if(LINUX)
    add_executable(NoisyCirclePathGenerationMSegment_RaycastVolume
        NoisyCirclePathGenerationMSegment_RaycastVolume.cpp
    )

    target_compile_features(NoisyCirclePathGenerationMSegment_RaycastVolume
        PUBLIC cxx_static_assert
        PUBLIC cxx_std_17
    )

    if(WIN32)
        target_compile_options(NoisyCirclePathGenerationMSegment_RaycastVolume
            PUBLIC "/openmp"
        )
    endif()

    target_link_libraries(NoisyCirclePathGenerationMSegment_RaycastVolume
        PUBLIC ExperimentBase
        PUBLIC Twisty
        PUBLIC glm
        PRIVATE nlohmann_json::nlohmann_json
        PUBLIC tinyexr_include
    )
# endif()

add_executable(Simple_FiveSegment_Analytical
    Simple_FiveSegment_Analytical.cpp
)

target_compile_features(Simple_FiveSegment_Analytical
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(Simple_FiveSegment_Analytical
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(Simple_FiveSegment_Analytical
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(Simple_SixSegment_Analytical
    Simple_SixSegment_Analytical.cpp
)

target_compile_features(Simple_SixSegment_Analytical
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(Simple_SixSegment_Analytical
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(Simple_SixSegment_Analytical
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)

add_executable(CalculateNormalizer
    CalculateNormalizer.cpp
)

target_compile_features(CalculateNormalizer
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(CalculateNormalizer
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(CalculateNormalizer
    PUBLIC ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PRIVATE nlohmann_json::nlohmann_json
)
