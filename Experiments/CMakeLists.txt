find_package(fmt CONFIG REQUIRED)

add_executable(BezierExperiment
    BezierExperiment.cpp
)

target_link_libraries(BezierExperiment
    Twisty
    fmt::fmt
    fmt::fmt-header-only
)

add_executable(IceCoreExperiment
    IceCoreExperiment.cpp
)

target_link_libraries(IceCoreExperiment
    Twisty
    fmt::fmt
    fmt::fmt-header-only
)

add_executable(PathWeightBehavior
    PathWeightBehavior.cpp
)

target_link_libraries(PathWeightBehavior
    Twisty
    fmt::fmt
    fmt::fmt-header-only
)