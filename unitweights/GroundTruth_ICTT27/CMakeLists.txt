add_executable(CalculateGroundTruth
    CalculateGroundTruth.cpp
)

target_include_directories(Twisty PUBLIC ${LIB_LIB_DIR})
target_link_libraries(Twisty PUBLIC ${LIB_LIB_LIBRARY})

target_link_libraries(CalculateGroundTruth
    PUBLIC FMath
    PUBLIC Twisty
    PRIVATE nlohmann_json::nlohmann_json
)

target_compile_features(CalculateGroundTruth
    PRIVATE cxx_std_17
)