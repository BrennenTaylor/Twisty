include(FetchContent)

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

FetchContent_Declare(tinyexr
    GIT_REPOSITORY "https://github.com/syoyo/tinyexr"
    GIT_TAG "release"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(tinyexr)

add_library(tinyexr_include INTERFACE)
target_include_directories(tinyexr_include INTERFACE ${tinyexr_SOURCE_DIR}/)
target_link_libraries(tinyexr_include INTERFACE miniz)

add_executable(FFE_ImageGen
    FFE_ImageGen.cpp
)

target_link_libraries(FFE_ImageGen
    PUBLIC Twisty
    PUBLIC tinyexr_include
)