include(FetchContent)
include(ExternalProject)

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