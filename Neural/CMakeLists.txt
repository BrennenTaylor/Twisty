add_subdirectory(dependencies)

add_executable(raymarch_proof
    raymarch_proof.cpp
)

target_link_libraries(raymarch_proof
    PUBLIC FMath::FMath
    PUBLIC tinyexr_include
)

set_property(TARGET raymarch_proof PROPERTY CXX_STANDARD 20)