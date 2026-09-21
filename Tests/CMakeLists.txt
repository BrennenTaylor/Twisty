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

include(CTest)
if(BUILD_TESTING)
    add_test(NAME WeightTableTest COMMAND WeightTableTest)
endif()
