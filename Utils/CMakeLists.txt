find_package(RapidJSON CONFIG REQUIRED)

add_executable(PathBatchWeightor
    PathBatchWeightor.cpp
)

target_link_libraries(PathBatchWeightor
    PUBLIC Twisty
)

target_compile_options(PathBatchWeightor
    PRIVATE /openmp
)

target_include_directories(PathBatchWeightor
    PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
)

# TODO: I dont believe these are needed anymore
# # PathSimilarityScoreUtil
# add_executable(PathSimilarityScoreUtil
#     PathSimilarityScoreUtil.cpp
# )

# target_link_libraries(PathSimilarityScoreUtil
#     PUBLIC Twisty
# )

# target_compile_options(PathSimilarityScoreUtil
#     PRIVATE /openmp
# )

# target_include_directories(PathSimilarityScoreUtil
#     PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
# )

# # GeometricCurvePurturb
# add_executable(GeometricCurvePurturb
#     GeometricCurvePurturb.cpp
# )

# target_link_libraries(GeometricCurvePurturb
#     PUBLIC Twisty
# )

# target_compile_options(GeometricCurvePurturb
#     PRIVATE /openmp
# )

# target_include_directories(GeometricCurvePurturb
#     PRIVATE ${RAPIDJSON_INCLUDE_DIRS}
# )