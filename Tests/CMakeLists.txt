# set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_VERBOSE_MAKEFILE on )

add_executable(BezierTest
    BezierTests.cpp
)

target_link_libraries(BezierTest
    Twisty
)

add_executable(PathWeightTest
    PathWeightTest.cpp
)

target_link_libraries(PathWeightTest
    Twisty
)

add_executable(GaussianFunctionTest
    GaussianFunctionTest.cpp
)

target_link_libraries(GaussianFunctionTest
    Twisty
)


add_executable(WeightSamplingIndependenceTest
    WeightSamplingIndependenceTest.cpp
)

target_link_libraries(WeightSamplingIndependenceTest
    Twisty
)