cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(BOOST REQUIRED)
find_package(FMath)
find_package(fmt CONFIG REQUIRED)
find_package(OpenEXR)

add_executable(FFEImageGen
    FFE_ImageGen.cpp
)

target_include_directories(FFEImageGen
    PUBLIC ${OpenEXR_INCLUDE_DIRS}
)

target_link_libraries(FFEImageGen
    PRIVATE FMath::FMath
    PRIVATE ${OpenEXR_LIBRARIES}
    PRIVATE fmt::fmt
    PRIVATE fmt::fmt-header-only
)

target_compile_definitions(FFEImageGen
    PUBLIC OPENEXR_DLL
)

add_executable(LogifyGridOfData
    LogifyGridOfData.cpp
)

target_include_directories(LogifyGridOfData
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(LogifyGridOfData
    PRIVATE FMath::FMath
    PRIVATE fmt::fmt
    PRIVATE fmt::fmt-header-only
)

add_executable(WeightSliceExperiment
    WeightSliceExperiment.cpp
)

target_include_directories(WeightSliceExperiment
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(WeightSliceExperiment
    PRIVATE FMath::FMath
    PRIVATE fmt::fmt
    PRIVATE fmt::fmt-header-only
)

add_executable(ConvergenceExperiment
    ConvergenceExperiment.cpp
)

target_include_directories(ConvergenceExperiment
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ConvergenceExperiment
    PRIVATE FMath::FMath
    PRIVATE fmt::fmt
    PRIVATE fmt::fmt-header-only
)

add_executable(ZeroToOneNormalization
    ZeroToOneNormalization.cpp
)

target_include_directories(ZeroToOneNormalization
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ZeroToOneNormalization
    PRIVATE FMath::FMath
    PRIVATE fmt::fmt
    PRIVATE fmt::fmt-header-only
)

add_executable(ExportedPathFixer
    ExportedPathFixer.cpp
)

target_include_directories(ExportedPathFixer
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ExportedPathFixer
    PUBLIC Twisty
)

add_executable(ExtractProblemPaths
    ExtractProblemPaths.cpp
)

target_include_directories(ExtractProblemPaths
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(ExtractProblemPaths
    PUBLIC Twisty
)

add_executable(SpringLengthExperiment
    SpringLengthExperiment.cpp
)

target_include_directories(SpringLengthExperiment
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(SpringLengthExperiment
    PUBLIC Twisty
)

add_executable(SolidAngleExperiment
    SolidAngleExperiment.cpp
)

target_include_directories(SolidAngleExperiment
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(SolidAngleExperiment
    PUBLIC Twisty
)

add_executable(PathHeatmap
    PathHeatmap.cpp
)

target_include_directories(PathHeatmap
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(PathHeatmap
    PUBLIC Twisty
)