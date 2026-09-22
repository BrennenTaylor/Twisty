# Helper functions for Twisty builds.
#
# twisty_add_experiment(<name> <sources...> [EXPERIMENT_BASE] [STB])
#   - links ExperimentBase (brings glm, tinyexr, miniz, OpenVDB) with EXPERIMENT_BASE,
#     else links Twisty (brings glm, nlohmann_json via Twisty PUBLIC).
#   - adds the stb/ (stb_image_write) include dir with STB.
#   - adds /openmp on WIN32.
#
# twisty_add_tool(<name> <sources...> [EXPERIMENT_BASE])
#   - links Twisty; additionally ExperimentBase with EXPERIMENT_BASE.

function(twisty_add_experiment target)
    cmake_parse_arguments(ARG "EXPERIMENT_BASE;STB" "" "" "${ARGN}")
    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})
    target_compile_features(${target} PUBLIC cxx_static_assert cxx_std_17)
    if(ARG_EXPERIMENT_BASE)
        target_link_libraries(${target} PUBLIC ExperimentBase)
    else()
        target_link_libraries(${target} PUBLIC Twisty)
    endif()
    if(ARG_STB)
        target_link_libraries(${target} PRIVATE stb_headers)
    endif()
    if(WIN32)
        target_compile_options(${target} PRIVATE /openmp)
    endif()
endfunction()

function(twisty_add_tool target)
    cmake_parse_arguments(ARG "EXPERIMENT_BASE" "" "" "${ARGN}")
    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries(${target} PUBLIC Twisty)
    if(ARG_EXPERIMENT_BASE)
        target_link_libraries(${target} PUBLIC ExperimentBase)
    endif()
endfunction()
