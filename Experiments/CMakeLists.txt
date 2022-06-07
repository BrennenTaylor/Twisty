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
    PRIVATE nlohmann_json::nlohmann_json
)

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