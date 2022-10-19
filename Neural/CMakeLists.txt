add_executable(raymarch_proof
    raymarch_proof.cpp
)

target_link_libraries(raymarch_proof
    PUBLIC FMath::FMath
)

set_property(TARGET raymarch_proof PROPERTY CXX_STANDARD 20)