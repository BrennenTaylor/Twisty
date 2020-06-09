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

add_executable(PathSetGen
    PathSetGen.cpp
)

target_link_libraries(PathSetGen
    Twisty
)

# Curve Error Tests
add_executable(CurveErrorFunctionTest
    CurveErrorFunctionTest.cpp
)

target_link_libraries(CurveErrorFunctionTest
    Twisty
)

add_executable(CurveCurvatureTest
CurveCurvatureTest.cpp
)

target_link_libraries(CurveCurvatureTest
    Twisty
)

add_executable(CurveTorsionTest
CurveTorsionTest.cpp
)

target_link_libraries(CurveTorsionTest
    Twisty
)

add_executable(CurvatureTorsionFrameTest
    CurvatureTorsionFrameTest.cpp
)

target_link_libraries(CurvatureTorsionFrameTest
    Twisty
)

add_executable(RootSolveTest
    RootSolveTest.cpp
)

target_link_libraries(RootSolveTest
    Twisty
    cminpack
)

# cuda_add_executable(RootSolveTestCuda
#     RootSolveTestCuda.cu
# )

# target_link_libraries(RootSolveTestCuda
#     cminpack
# )