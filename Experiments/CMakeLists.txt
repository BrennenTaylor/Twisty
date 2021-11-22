add_executable(FullExperiment
    FullExperiment.cpp
)

target_compile_features(FullExperiment
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

target_link_libraries(FullExperiment
    PUBLIC Twisty
    PUBLIC glm
    PUBLIC libconfig++
    PUBLIC shlwapi
)

# add_executable(SmallSegmentExperiment
#     SmallSegmentExperiment.cpp
# )

# target_link_libraries(SmallSegmentExperiment
#     Twisty
# )