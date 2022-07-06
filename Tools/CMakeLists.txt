cmake_minimum_required(VERSION 3.10)

# if (WIN32)
# message("WIN32 Defined")
# FetchContent_Declare(tinyexr
#     GIT_REPOSITORY "https://github.com/syoyo/tinyexr"
# )
# FetchContent_MakeAvailable(tinyexr)
# target_include_directories(tinyexr INTERFACE ${tinyexr_SOURCE_DIR})
# endif(WIN32)

# IF (WIN32)
# add_executable(FFEImageGen
#     FFE_ImageGen.cpp
# )

# target_link_libraries(FFEImageGen
#     PRIVATE FMath::FMath
#     PRIVATE tinyexr
# )
# ENDIF()
# add_executable(LogifyAndMinMaxData
#     LogifyAndMinMaxData.cpp
# )

# target_include_directories(LogifyAndMinMaxData
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(LogifyAndMinMaxData
#     PRIVATE FMath::FMath
# )

# add_executable(MinMaxGivenVals
#     MinMaxGivenVals.cpp
# )

# target_include_directories(MinMaxGivenVals
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(MinMaxGivenVals
#     PRIVATE FMath::FMath
# )

# add_executable(GenerateFFDistances
#     GenerateFFDistances.cpp
# )

# target_include_directories(GenerateFFDistances
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(GenerateFFDistances
#     PRIVATE FMath::FMath
# )

# add_executable(GenerateFFDistancesHalf
#     GenerateFFDistancesHalf.cpp
# )

# target_include_directories(GenerateFFDistancesHalf
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(GenerateFFDistancesHalf
#     PRIVATE FMath::FMath
# )

# add_executable(WeightSliceExperiment
#     WeightSliceExperiment.cpp
# )

# target_include_directories(WeightSliceExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(WeightSliceExperiment
#     PRIVATE FMath::FMath
# )

# add_executable(ConvergenceExperiment
#     ConvergenceExperiment.cpp
# )

# target_compile_features(ConvergenceExperiment
#     PUBLIC cxx_static_assert
#     PUBLIC cxx_std_17
# )

# target_include_directories(ConvergenceExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(ConvergenceExperiment
#     PRIVATE FMath::FMath
# )

# add_executable(HighestWeightPathExperiment
#     HighestWeightPathExperiment.cpp
# )

# target_include_directories(HighestWeightPathExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(HighestWeightPathExperiment
#     PRIVATE FMath::FMath
# )

# add_executable(ZeroToOneNormalization
#     ZeroToOneNormalization.cpp
# )

# target_include_directories(ZeroToOneNormalization
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(ZeroToOneNormalization
#     PRIVATE FMath::FMath
# )

add_executable(ExportedPathFixer
    ExportedPathFixer.cpp
)

target_include_directories(ExportedPathFixer
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ExportedPathFixer
    PUBLIC Twisty
    PUBLIC nlohmann_json::nlohmann_json
)

add_executable(ExportedPathWeightsFixer
    ExportedPathWeightsFixer.cpp
)

target_include_directories(ExportedPathWeightsFixer
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ExportedPathWeightsFixer
    PUBLIC Twisty
    PUBLIC nlohmann_json::nlohmann_json
)

add_executable(ExportedFiveSegmentsFixer
    ExportedFiveSegmentsFixer.cpp
)

target_include_directories(ExportedFiveSegmentsFixer
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ExportedFiveSegmentsFixer
    PUBLIC Twisty
    PUBLIC nlohmann_json::nlohmann_json
)

# add_executable(ExtractProblemPaths
#     ExtractProblemPaths.cpp
# )

# target_include_directories(ExtractProblemPaths
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(ExtractProblemPaths
#     PUBLIC Twisty
# )

# add_executable(SpringLengthExperiment
#     SpringLengthExperiment.cpp
# )

# target_include_directories(SpringLengthExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(SpringLengthExperiment
#     PUBLIC Twisty
# )

# add_executable(SolidAngleExperiment
#     SolidAngleExperiment.cpp
# )

# target_include_directories(SolidAngleExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(SolidAngleExperiment
#     PUBLIC Twisty
# )

# add_executable(CurvatureHistogramExperiment
#     CurvatureHistogramExperiment.cpp
# )

# target_include_directories(CurvatureHistogramExperiment
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(CurvatureHistogramExperiment
#     PUBLIC Twisty
# )

# add_executable(Sandbox
#     Sandbox.cpp
# )

# target_include_directories(Sandbox
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(Sandbox
#     PUBLIC Twisty
# )

# add_executable(Sandbox2
#     Sandbox2.cpp
# )

# target_include_directories(Sandbox2
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(Sandbox2
#     PUBLIC Twisty
# )

# add_executable(InitialCurveGenerator
#     InitialCurveGenerator.cpp
# )

# target_include_directories(InitialCurveGenerator
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(InitialCurveGenerator
#     PUBLIC Twisty
# )

# add_executable(CurveWeightor
#     CurveWeightor.cpp
# )

# target_include_directories(CurveWeightor
#     PUBLIC ${Boost_INCLUDE_DIR}
# )

# target_link_libraries(CurveWeightor
#     PUBLIC Twisty
# )
