add_executable(WeightTableTest
    WeightTableTest.cpp
)

target_compile_features(WeightTableTest
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(WeightTableTest
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(WeightTableTest
    PUBLIC Twisty
)

add_executable(TwoSegmentResolveTest
    TwoSegmentResolveTest.cpp
)

target_compile_features(TwoSegmentResolveTest
    PUBLIC cxx_std_17
)

target_link_libraries(TwoSegmentResolveTest
    PUBLIC Twisty
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME WeightTableTest COMMAND WeightTableTest)
    set_tests_properties(WeightTableTest PROPERTIES TIMEOUT 60)

    add_test(NAME TwoSegmentResolveTest COMMAND TwoSegmentResolveTest)
    set_tests_properties(TwoSegmentResolveTest PROPERTIES TIMEOUT 60)
endif()
