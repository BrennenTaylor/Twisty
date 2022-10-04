add_executable(ExportedPathFixer
    ExportedPathFixer.cpp
)

target_link_libraries(ExportedPathFixer
    PUBLIC Twisty
)

add_executable(ExportedPathWeightsFixer
    ExportedPathWeightsFixer.cpp
)

target_link_libraries(ExportedPathWeightsFixer
    PUBLIC Twisty
)

add_executable(ExportedFiveSegmentsFixer
    ExportedFiveSegmentsFixer.cpp
)

target_link_libraries(ExportedFiveSegmentsFixer
    PUBLIC Twisty
)