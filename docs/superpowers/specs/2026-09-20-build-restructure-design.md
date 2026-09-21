# Twisty Build Restructure — Design

**Date:** 2026-09-20
**Status:** Approved in chat (sections 1–5)
**Author:** opencode (with Brennen Taylor)

## Context

Twisty is a research codebase for multiple-scattering path-tracing.
Its build today is a working-but-legacy CMake setup that evolved from a
Windows-first project. The Linux build was just made to work
(PR #1), exposing structural debt:

- 491-line `Experiments/CMakeLists.txt` repeating the same
  `add_executable` + link boilerplate ~23 times.
- `find_package` for OpenVDB / nlohmann_json duplicated across
  subdirectories; a dangling `Boost_INCLUDE_DIR` reference with no
  `find_package(Boost)`.
- C++17 standard set only for non-Linux builds.
- `Tests/` built only on Windows; no CTest registration.
- A stale `Neural/` standalone build referencing a deleted
  `tinyexr_include` target (would not configure).
- Three stray experiment `.cpp` files at the repo root never built.
- Two git submodules (`openvdb` — already removed — and `stb`) with the
  associated gitlink friction.

## Goals

1. A CMake structure "as it would be built today": one source of truth
   for compiler/standard settings, a single dependency owner, helper
   functions that remove boilerplate, presets, and CTest.
2. Linux-first, but Windows source files (Utils/, Viewer/, unitweights/,
   RelatedWork/) stay in place and dormant.
3. Dependencies managed via apt system packages (OpenVDB, glm,
   nlohmann-json, Boost, TBB, OpenMP) plus FetchContent (FMath, stb).
4. One bootstrap script; README documents the build.
5. No git submodules.

## Non-goals

- No change to directory layout (`src/...`), library internals,
  `ExperimentBase` contents, or any source code.
- No rework of the CUDA runner semantics (only re-gating of
  `enable_language(CUDA)`).
- No Windows build hardening beyond keeping existing MSVC presets
  functional.
- `PhaseFunctions/`, `Tools/deprecated/` remain unbuilt and untouched.

## Decisions (from Q&A)

| Question | Decision |
|---|---|
| Platform scope | Linux-first; leave Windows files dormant |
| `Neural/` | Stub out: delete its CMake build scaffolding, keep sources |
| Dependency management | apt + bootstrap script |
| Tests | Enable CTest on Linux |
| Stray root `.cpps` | Move into `Experiments/`, wire into build |
| Submodules | Remove both `stb` and `openvdb`; no submodules remain |
| stb handling | FetchContent (consistent with FMath), not vendored |

## Proposed Structure

### Section 1 — Root `CMakeLists.txt` + presets

Root becomes configuration-only:

```cmake
cmake_minimum_required(VERSION 3.20)
project(Twisty VERSION 1.0.0 LANGUAGES CXX)

include(cmake/TwistyHelpers.cmake)

option(USE_CUDA "Enable Cuda and GPU runner?" FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(WIN32)   # was: if (NOT LINUX)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # VS globals: UseMultiToolTask / EnforceProcessCountAcrossBuilds
endif()

add_subdirectory(dependencies)
add_subdirectory(Twisty)
add_subdirectory(Tools)
add_subdirectory(Experiments)

enable_testing()
add_subdirectory(Tests)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "52;70;72")
endif()
```

Notes:
- `enable_testing()` + unconditional `add_subdirectory(Tests)`.
- CUDA arch list moves from `Twisty/CMakeLists.txt` to root.
- On Linux the build remains shared (Twisty.so, ExperimentBase.so) —
  current behavior is preserved; we do not force `BUILD_SHARED_LIBS`
  on Linux.

**CMakePresets.json** — add:

```json
{
  "name": "linux",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "RelWithDebInfo",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
  }
}
```

Build path: `cmake --preset linux && cmake --build build`.
Existing MSVC presets are retained unchanged.

### Section 2 — Helper functions and target collapse

New `cmake/TwistyHelpers.cmake`:

```cmake
function(twisty_add_experiment target)
    cmake_parse_arguments(ARG "" "" "EXPERIMENT_BASE;STB" "${ARGN}")
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
endfunction()

function(twisty_add_tool target)
    cmake_parse_arguments(ARG "" "" "EXPERIMENT_BASE" "${ARGN}")
    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries(${target} PUBLIC Twisty)
    if(ARG_EXPERIMENT_BASE)
        target_link_libraries(${target} PUBLIC ExperimentBase)
    endif()
endfunction()
```

Rationale: every experiment links `PUBLIC Twisty` today; `Twisty`
already PUBLIC-links glm and nlohmann_json, so the extra
`PUBLIC glm` / `PRIVATE nlohmann_json` lines are redundant
(proven by `FullExperiment` / `StressTestCombinedWeights`, which only
link `Twisty`).

**`Experiments/CMakeLists.txt`** — collapses from ~491 lines to ~85:
`ExperimentBase` definition (with the current tinyexr/miniz wiring
unchanged) + one `twisty_add_experiment(...)` line per executable.
Existing target names and `EXPERIMENT_BASE` / `STB` flags mapped from
the current link blocks.

**`Tools/CMakeLists.txt`** — four one-liners
(`FFE_ImageGen` uses the `EXPERIMENT_BASE` flag).

### Section 3 — Single dependency owner + bootstrap

**`dependencies/CMakeLists.txt`** becomes the only place third-party
deps are discovered:

```cmake
include(FetchContent)

# --- Source-built (FetchContent), version-controlled ---
FetchContent_Declare(FMath GIT_REPOSITORY https://github.com/BrennenTaylor/FMath)
FetchContent_MakeAvailable(FMath)

FetchContent_Declare(stb GIT_REPOSITORY https://github.com/nothings/stb GIT_TAG <pinned-tag>)
FetchContent_MakeAvailable(stb)
# stb provides no CMake target: create the interface used by TwistyHelpers STB flag
if(NOT TARGET stb_headers)
    add_library(stb_headers INTERFACE)
    target_include_directories(stb_headers INTERFACE ${stb_SOURCE_DIR})
endif()

# --- System (apt) deps ---
find_package(OpenMP REQUIRED)
find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED)   # header-only use; no components today
if(LINUX)
    find_package(OpenVDB REQUIRED)   # via /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/OpenVDB
endif()
```

- `Twisty/CMakeLists.txt` and `Experiments/CMakeLists.txt` drop their
  local `find_package` calls (and the `CMAKE_MODULE_PATH` OpenVDB block is
  moved here). Targets keep linking by name (`OpenMP::OpenMP_CXX`,
  `OpenVDB::openvdb`, `glm`, ...).
- The dead `${Boost_INCLUDE_DIR}` include in `Twisty/CMakeLists.txt` is
  replaced with `Boost::boost` if Twisty consumes Boost headers, else
  dropped.
- `include(ExternalProject)` removed (unused).

**`scripts/bootstrap-ubuntu.sh`** (new, executable):

```bash
#!/usr/bin/env bash
set -euo pipefail
sudo apt update && sudo apt install -y \
    build-essential cmake ninja-build pkg-config \
    libglm-dev nlohmann-json3-dev libopenvdb-dev libtbb-dev libomp-dev libboost-dev
```

No submodule step (submodules removed).

**`README.md`** — replace stale `install.txt` content with:

```
./scripts/bootstrap-ubuntu.sh
cmake --preset linux && cmake --build build
ctest --test-dir build
```

`install.txt` removed (contents folded into README).

### Section 4 — Tests (CTest)

**`Tests/CMakeLists.txt`**:

```cmake
add_executable(WeightTableTest WeightTableTest.cpp)
target_compile_features(WeightTableTest PUBLIC cxx_static_assert cxx_std_17)
target_link_libraries(WeightTableTest PUBLIC Twisty)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME WeightTableTest COMMAND WeightTableTest)
endif()
```

`Tests/` is removed from the `WIN32` gate. `ninja test` runs it.

### Section 5 — Neural, stray files, cleanup

- **Neural/:** delete `Neural/CMakeLists.txt`,
  `Neural/CMakePresets.json`, `Neural/dependencies/`. Keep sources.
- **Stray root `.cpps` → Experiments/:**
  - `Benchmark_5_1.cpp` → `twisty_add_experiment(Benchmark_5_1 ...)` (plain)
  - `DifferentNormalSThetaExperiment.cpp` → plain
  - `FullExperiment_CombinedInitialCurves.cpp` → moved into
    `Experiments/` but **NOT added to the build** — it includes
    `FullExperimentRunnerOptimalPerturbOptimized.h`, which does not
    exist. **Open question** (see Risks).
- **Delete `.gitmodules`** (no submodules remain after stb removal).
- `Utils/`, `Viewer/`, `unitweights/`, `RelatedWork/`, `Tools/deprecated/`
  remain unbuilt/dormant. `.gitignore` build rules stay as-is.

## File Change Inventory

| File | Action |
|---|---|
| `CMakeLists.txt` | Rewrite (runtime-only root) |
| `CMakePresets.json` | Add `linux` preset |
| `cmake/TwistyHelpers.cmake` | New |
| `dependencies/CMakeLists.txt` | Rewrite as single dep owner |
| `Twisty/CMakeLists.txt` | Remove find_package / CUDA gate; Boost include fix |
| `Experiments/CMakeLists.txt` | Collapse to ~85 lines via helper |
| `Tools/CMakeLists.txt` | Collapse via helper |
| `Tests/CMakeLists.txt` | Build everywhere + CTest registration |
| `scripts/bootstrap-ubuntu.sh` | New |
| `README.md` | Rewrite build docs |
| `install.txt` | Remove |
| `.gitmodules` | Delete |
| `stb/` (submodule) | Remove from tree & index |
| `openvdb` | Already removed in PR #1 |
| `Neural/CMakeLists.txt`, `Neural/CMakePresets.json`, `Neural/dependencies/` | Delete |
| `Benchmark_5_1.cpp`, `DifferentNormalSThetaExperiment.cpp` | Move → `Experiments/`, wire in |
| `FullExperiment_CombinedInitialCurves.cpp` | Move → `Experiments/`, unbuilt (see Risks) |

## Verification

1. `./scripts/bootstrap-ubuntu.sh`
2. `cmake --preset linux`
3. `cmake --build build` (must compile from clean; ninja "no work to do")
4. `ctest --test-dir build` (WeightTableTest passes)
5. `git submodule status` → empty / no `.gitmodules` errors
6. Confirm existing MSVC presets still reference valid targets
   (configure-and-generate check on a Windows host or trusted review)

## Risks / Open Questions

1. **`FullExperiment_CombinedInitialCurves.cpp`** cannot compile:
   includes a nonexistent header. Options: (a) treat as dead, move +
   don't build; (b) investigate whether it should target the existing
   `FullExperimentRunnerOptimalPerturb.h`; (c) delete.
   **Decision needed before implementation.**
2. **glm** is currently resolved transitively from FMath's FetchContent,
   not apt. We explicitly `find_package(glm)` — need to confirm no
   target conflict when both exist (expected: CMake dedupes the glm
   target).
3. **Boost** — Twisty includes Boost headers today with no
   `find_package`; declaring it and adding `Boost::boost` include dirs
   must reproduce current behavior (header-only usage).
4. **stb tag** — pick a recent pinned tag at implementation time.
5. Neural builds are excluded; if `raymarch_vdb.cpp` etc. are needed
   for active research, flag during implementation.

## Transition to Implementation

Next: use the writing-plans skill to produce a step-by-step plan from
this design, then execute with a clean-build verification checkpoint at
the end (and after any spins on risk items 1–4).