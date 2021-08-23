add_executable(CalculateGroundTruth
    CalculateGroundTruth.cpp
)

target_link_libraries(CalculateGroundTruth
    PUBLIC FMath
)

target_compile_features(CalculateGroundTruth
    PRIVATE cxx_std_17
)