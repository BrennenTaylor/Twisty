project(Fred2017)

add_executable(Fred_2017_Benchmark_5_1
    Fred_2017_Benchmark_5_1.cpp
)

target_link_libraries(Fred_2017_Benchmark_5_1
    PUBLIC FMath::FMath
    PUBLIC Twisty
)

target_compile_features(Fred_2017_Benchmark_5_1
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)