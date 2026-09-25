include(FetchContent)

# --- Version-controlled source dependencies (FetchContent) ---
FetchContent_Declare(glm
    GIT_REPOSITORY "https://github.com/g-truc/glm.git"
    GIT_TAG "1.0.3"
)
FetchContent_MakeAvailable(glm)

# stb is a set of stable single-file libraries. Pin to the exact commit
# the removed stb submodule was at. Its tree has no CMakeLists.txt, so
# MakeAvailable only populates it.
FetchContent_Declare(stb
    GIT_REPOSITORY "https://github.com/nothings/stb"
    GIT_TAG "af1a5bc352164740c1cc1354942b1c6b72eacb8a"
)
FetchContent_MakeAvailable(stb)

if(NOT TARGET stb_headers)
    add_library(stb_headers INTERFACE)
    target_include_directories(stb_headers INTERFACE ${stb_SOURCE_DIR})
endif()

# --- System (apt) dependencies, discovered once ---
find_package(OpenMP REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED)

if(LINUX)
    list(APPEND CMAKE_MODULE_PATH
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/OpenVDB"
    )
    find_package(OpenVDB REQUIRED)
endif()

# find_package creates non-GLOBAL IMPORTED targets, which are only visible in
# this directory and its subdirectories -- not in sibling subdirectories such
# as Twisty/. Promote them to GLOBAL so the single dependency owner works.
# Any future find_package added here MUST be added to this promotion too, or its imported targets stay invisible to sibling subdirectories.
set_target_properties(OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
    Boost::headers
    PROPERTIES IMPORTED_GLOBAL TRUE
)
if(LINUX)
    set_target_properties(OpenVDB::openvdb PROPERTIES IMPORTED_GLOBAL TRUE)
endif()