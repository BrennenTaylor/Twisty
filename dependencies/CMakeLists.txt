include(FetchContent)
include(ExternalProject)

# FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
# FetchContent_MakeAvailable(json)

FetchContent_Declare(FMath
    GIT_REPOSITORY "https://github.com/BrennenTaylor/FMath"
)
FetchContent_MakeAvailable(FMath)