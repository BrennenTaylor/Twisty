add_executable(WeightExplorer
    WeightExplorer.cpp
)

target_link_libraries(WeightExplorer
    PUBLIC Twisty
)

# add_executable(PathBatchWeightor
#     PathBatchWeightor.cpp
# )

# target_link_libraries(PathBatchWeightor
#     PUBLIC Twisty
# )

# target_compile_options(PathBatchWeightor
#     PRIVATE /openmp
# )