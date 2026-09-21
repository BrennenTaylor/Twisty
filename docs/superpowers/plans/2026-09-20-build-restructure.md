# Build Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Twisty's CMake build into a modern, Linux-first structure: helper-function-driven targets, a single dependency owner, CTest, presets, a bootstrap script, and zero submodules.

**Architecture:** Configuration stays a thin root `CMakeLists.txt`; shared target boilerplate moves into `cmake/TwistyHelpers.cmake` (new `twisty_add_experiment` / `twisty_add_tool` functions); `dependencies/` becomes the only place third-party deps are discovered (`find_package` for apt packages, `FetchContent` for FMath/stb/tinyexr); each task keeps the build green and verifiable before committing.

**Tech Stack:** CMake >= 3.20, Ninja, C++17. Linux (Ubuntu) primary; Windows build files kept dormant. ctest for tests.

**Spec:** `docs/superpowers/specs/2026-09-20-build-restructure-design.md`

## Global Constraints

- CMake minimum 3.20; project `Twisty VERSION 1.0.0 LANGUAGES CXX`.
- C++17, required, no extensions: `CMAKE_CXX_STANDARD 17`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`.
- Build dir is `build/`, generator Ninja, preset `linux` (RelWithDebInfo, export compile commands).
- Dependencies: apt `find_package` (OpenMP, nlohmann_json, Boost; OpenVDB on Linux) + `FetchContent` only for FMath, stb (pinned `af1a5bc352164740c1cc1354942b1c6b72eacb8a`), tinyexr (`release`).
- glm is **not** `find_package`d — it comes transitively from FMath's own FetchContent (adding a second caller would create a duplicate `glm` target and break configure). Verify glm is still linked by name `glm` exactly as today.
- No git submodules remain after this plan. `stb` and `tinyexr` gitlinks plus `.gitmodules` are removed; `openvdb` gitlink already removed.
- All experiment executables use `twisty_add_experiment(...)`; all tool executables use `twisty_add_tool(...)`.
- `FullExperiment_CombinedInitialCurves.cpp` is moved into `Experiments/` but **not added** to the build (includes a nonexistent header).
- Windows behavior change (accepted, EDGE-1): all experiments get `/openmp` on WIN32 via the helper (previously only some did). Dormant platform; OpenMP is already linked through `Twisty` on all platforms.
- `FetchContent_Populate` deprecation warning for tinyexr in `Experiments/` is accepted (proven pattern; upstream tinyexr's own build is broken under `-Weverything -Werror`, so `FetchContent_MakeAvailable` is intentionally avoided there).
- Every task ends with a green verification: `cmake --preset linux` + `cmake --build build` (+ `ctest` from Task 5 on).

---

### Task 1: Root CMakeLists + help helper functions + linux preset

**Files:**
- Create: `cmake/TwistyHelpers.cmake`
- Rewrite: `CMakeLists.txt`
- Modify: `CMakePresets.json`

**Interfaces:**
- Consumes: nothing new (functions are defined but unused until Tasks 3–4).
- Produces: functions `twisty_add_experiment(<name> <sources...> [EXPERIMENT_BASE] [STB])` and `twisty_add_tool(<name> <sources...> [EXPERIMENT_BASE])`; CMake preset named `linux`; root project settings (C++17, `LINUX`, `WIN32` gates, tests scaffolding, CUDA gate).

- [ ] **Step 1: Create `cmake/TwistyHelpers.cmake`**

```cmake
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
    if(WIN32)
        target_compile_options(${target} PRIVATE /openmp)
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

- [ ] **Step 2: Rewrite `CMakeLists.txt`** (full replacement)

```cmake
cmake_minimum_required(VERSION 3.20)

project(Twisty VERSION 1.0.0 LANGUAGES CXX)

include(cmake/TwistyHelpers.cmake)

# Needed for GPU build
option(USE_CUDA "Enable Cuda and GPU runner?" FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(WIN32)
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)UseMultiToolTask=")
        list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    endif()

    if(NOT CMAKE_VS_GLOBALS MATCHES "(^|;)EnforceProcessCountAcrossBuilds=")
        list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
    endif()
endif()

add_subdirectory(dependencies)
add_subdirectory(Twisty)
add_subdirectory(Tools)
add_subdirectory(Experiments)

enable_testing()
if(WIN32)
    add_subdirectory(Tests)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "52;70;72")
endif()
```

Note: `Tests/` stays WIN32-gated in this task; Task 5 un-gates it. CUDA arch list moved here from `Twisty/CMakeLists.txt`; the `Twisty` CUDA block retains its compile-option handling (Task 2 removes only the stale references).

- [ ] **Step 3: Add the `linux` preset to `CMakePresets.json`**

Inside the `"configurePresets": [ ... ]` array, append:

```json
        {
            "name": "linux",
            "displayName": "Ninja w/ Linux - amd64",
            "description": "Ninja using compilers for Linux (x64 architecture)",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "RelWithDebInfo",
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        }
```

(The existing MSVC presets must remain untouched.)

- [ ] **Step 4: Verify configure + build are green**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
```

Expected: configure succeeds, full build succeeds (same targets as before this task — helper functions are defined but unused yet; root settings only). Ignore the known `FetchContent_Populate(tinyexr)` deprecation warning.

- [ ] **Step 5: Commit**

```bash
git add cmake/TwistyHelpers.cmake CMakeLists.txt CMakePresets.json
git commit -m "build: centralize repo config and add twisty target helpers"
```

---

### Task 2: Single dependency owner + Twisty cleanup

**Files:**
- Rewrite: `dependencies/CMakeLists.txt`
- Modify: `Twisty/CMakeLists.txt`

**Interfaces:**
- Consumes: the root settings from Task 1 (`LINUX` var).
- Produces: `stb_headers` INTERFACE target (include dir = stb fetch); `FMath::FMath`, `glm`, `nlohmann_json::nlohmann_json`, `OpenMP::OpenMP_CXX`, `Boost::headers`, `OpenVDB::openvdb` available to all subdirectories; the old per-subdir `find_package` calls in `Twisty/CMakeLists.txt` removed.

- [ ] **Step 1: Rewrite `dependencies/CMakeLists.txt`** (full replacement)

```cmake
include(FetchContent)

# --- Version-controlled source dependencies (FetchContent) ---
FetchContent_Declare(FMath
    GIT_REPOSITORY "https://github.com/BrennenTaylor/FMath"
)
FetchContent_MakeAvailable(FMath)

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
```

Notes:
- glm is intentionally absent — FMath fetch-pulls glm and defines the singlar `glm` target. Do NOT add `find_package(glm)` here or anywhere (duplicate-target configure error).
- `include(ExternalProject)` is dropped (nothing uses it since tinyexr stopped being an external build).
- The `tinyexr` FetchContent stays in `Experiments/CMakeLists.txt` (Task 3 keeps it), untouched.

- [ ] **Step 2: Clean up `Twisty/CMakeLists.txt`**

Replace the top three blocks (current lines 1–11) — the `find_package(OpenMP)`, the `if(LINUX)` `CMAKE_MODULE_PATH`+`find_package(OpenVDB)` block, and `find_package(nlohmann_json)` — with nothing (that logic now lives in `dependencies/`).

In `target_link_libraries(Twisty ...)` add the Boost include target (replaces the dangling `${Boost_INCLUDE_DIR}` referenced in `target_include_directories`):

```cmake
target_link_libraries(Twisty
    PUBLIC FMath::FMath
    PUBLIC glm
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC Boost::headers
)
```

And in `target_include_directories(Twisty ...)` remove the `${Boost_INCLUDE_DIR}` line:

```cmake
target_include_directories(Twisty
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<$<BOOL:${USE_CUDA}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)
```

Keep everything else in the file unchanged (sources list, `shlwapi`, `OpenVDB::openvdb`, `JSON_DIAGNOSTICS=1`, CUDA block). Do not touch the CUDA arch list duplicated there (a `set_target_properties(... CUDA_ARCHITECTURES ...)` still applies to targets; the root `set(CMAKE_CUDA_ARCHITECTURES ...)` from Task 1 is harmless and consistent).

If `Boost::headers` is not found by `find_package(Boost)` (Boost < 1.70 or no config), fall back to linking `${Boost_INCLUDE_DIRS}` instead and note it in the commit message.

- [ ] **Step 3: Verify configure + build are green**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
```

Expected: succeeds with unchanged target set. Confirm `Twisty` still links (no unresolved `boost::` symbols at link time; Boost is header-only so expect none).

- [ ] **Step 4: Commit**

```bash
git add dependencies/CMakeLists.txt Twisty/CMakeLists.txt
git commit -m "build: consolidate dependency discovery into dependencies/ and declare Boost"
```

---

### Task 3: Collapse Experiments/CMakeLists.txt

**Files:**
- Rewrite: `Experiments/CMakeLists.txt`

**Interfaces:**
- Consumes: `twisty_add_experiment` (Task 1), `stb_headers` (Task 2), `ExperimentBase` and `tinyexr` FetchContent wiring preserved inline.
- Produces: the same 23 executables as today, via one helper call each.

- [ ] **Step 1: Rewrite `Experiments/CMakeLists.txt`** (full replacement)

```cmake
# tinyexr: fetched headers only; a single TINYEXR_IMPLEMENTATION TU
# (tinyexr_impl.cpp) compiles the implementation into ExperimentBase, and the
# bundled miniz is built as a small static lib to satisfy its deflate needs.
FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY "https://github.com/syoyo/tinyexr"
    GIT_TAG "release"
)

FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()

add_library(tinyexr_interface INTERFACE)
target_include_directories(tinyexr_interface INTERFACE ${tinyexr_SOURCE_DIR}
${tinyexr_SOURCE_DIR}/deps/miniz
)

enable_language(C)
add_library(miniz STATIC ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(miniz PUBLIC ${tinyexr_SOURCE_DIR}/deps/miniz)
set_property(TARGET miniz PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tinyexr_interface INTERFACE miniz)

add_library(ExperimentBase
    ExperimentBase.h
    ExperimentBase.cpp
    ExperimentUtils.h
    ExperimentUtils.cpp
    tinyexr_impl.cpp
)

target_include_directories(ExperimentBase INTERFACE .)

target_compile_features(ExperimentBase
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(ExperimentBase
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(ExperimentBase
    PUBLIC Twisty
    PUBLIC glm
    PUBLIC tinyexr_interface
    PRIVATE nlohmann_json::nlohmann_json
)

if(LINUX)
    target_link_libraries(ExperimentBase
        PUBLIC OpenVDB::openvdb
    )
endif()

# --- Experiments ---
twisty_add_experiment(FullExperiment                        FullExperiment.cpp)
twisty_add_experiment(StressTestCombinedWeights             StressTestCombinedWeights.cpp)
twisty_add_experiment(NoisyCircleAngleIntegration           NoisyCircleAngleIntegration.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCircleExperimentHalf             NoisyCircleExperimentHalf.cpp)
twisty_add_experiment(BeamSpreadExperiment                  BeamSpreadExperiment.cpp)
twisty_add_experiment(NoisyCircleExperiment                 NoisyCircleExperiment.cpp)
twisty_add_experiment(RingBenchmark                         RingBenchmark.cpp)
twisty_add_experiment(FiveSegmentExploreDoF                 FiveSegmentExploreDoF.cpp STB)
twisty_add_experiment(SixSegmentExploreDoF                  SixSegmentExploreDoF.cpp STB)
twisty_add_experiment(FiveSegmentHeatmap                    FiveSegmentHeatmap.cpp STB)
twisty_add_experiment(FiveSegmentAnglePathIntegral          FiveSegmentAnglePathIntegral.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentAnglePathIntegral           SixSegmentAnglePathIntegral.cpp EXPERIMENT_BASE)
twisty_add_experiment(FiveSegmentUniformPathGenerationSolver FiveSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentUniformPathGenerationSolver  SixSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(MSegmentUniformPathGenerationSolver    MSegmentUniformPathGenerationSolver.cpp EXPERIMENT_BASE)
twisty_add_experiment(FiveSegmentAngleSpaceMC               FiveSegmentAngleSpaceMC.cpp EXPERIMENT_BASE)
twisty_add_experiment(SixSegmentAngleSpaceMC                SixSegmentAngleSpaceMC.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment     NoisyCirclePathGenerationMSegment.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment_Pinhole NoisyCirclePathGenerationMSegment_Pinhole.cpp EXPERIMENT_BASE)
twisty_add_experiment(NoisyCirclePathGenerationMSegment_RaycastVolume NoisyCirclePathGenerationMSegment_RaycastVolume.cpp EXPERIMENT_BASE)
twisty_add_experiment(Simple_FiveSegment_Analytical         Simple_FiveSegment_Analytical.cpp EXPERIMENT_BASE)
twisty_add_experiment(Simple_SixSegment_Analytical          Simple_SixSegment_Analytical.cpp EXPERIMENT_BASE)
twisty_add_experiment(CalculateNormalizer                   CalculateNormalizer.cpp EXPERIMENT_BASE)
```

Removed vs current file: the local `find_package(OpenMP)` / `find_package(OpenVDB)` / `find_package(nlohmann_json)` blocks (now in `dependencies/`), the `PUBLIC tinyexr_interface` line on `NoisyCirclePathGenerationMSegment_RaycastVolume` (comes via `ExperimentBase`), and all per-target link/compile blocks. Note `FiveSegmentAnglePathIntegral`'s `WIN32 /openmp` and others are now covered by EDGE-1 (helper adds `/openmp` to all on WIN32).

- [ ] **Step 2: Verify configure + build are green**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
```

Expected: succeeds. Then confirm the binary count and names are unchanged:

```bash
ls build/Experiments | grep -v '\.' | sort > /tmp/after_experiments.txt
```

Compare to the executables that existed before this task (the 23 names listed above; files without a `.` in the name). Run `ninja -C build` a second time — must report "no work to do".

- [ ] **Step 3: Commit**

```bash
git add Experiments/CMakeLists.txt
git commit -m "build: replace experiment boilerplate with twisty_add_experiment"
```

---

### Task 4: Collapse Tools/CMakeLists.txt

**Files:**
- Rewrite: `Tools/CMakeLists.txt`

**Interfaces:**
- Consumes: `twisty_add_tool` (Task 1), `ExperimentBase` (Task 3).
- Produces: the same 5 tool executables, via one helper call each.

- [ ] **Step 1: Rewrite `Tools/CMakeLists.txt`** (full replacement)

```cmake
# --- Tools ---
twisty_add_tool(ExportedPathFixer             ExportedPathFixer.cpp)
twisty_add_tool(ExportedPathWeightsFixer      ExportedPathWeightsFixer.cpp)
twisty_add_tool(ExportedFiveSegmentsFixer     ExportedFiveSegmentsFixer.cpp)
twisty_add_tool(FFE_ImageGen                  FFE_ImageGen.cpp EXPERIMENT_BASE)
twisty_add_tool(WeightIntegralFunctionExplorer WeightIntegralFunctionExplorer.cpp)
```

(This replaces the current four `add_executable` blocks for `ExportedPathFixer`, `ExportedPathWeightsFixer`, `ExportedFiveSegmentsFixer`, `FFE_ImageGen`, `WeightIntegralFunctionExplorer` — `FFE_ImageGen` keeps its extra `ExperimentBase` link via the `EXPERIMENT_BASE` flag.)

- [ ] **Step 2: Verify configure + build are green**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
```

Expected: succeeds; `Tools/` shows the same 5 executables. Rerun `ninja -C build` → "no work to do".

- [ ] **Step 3: Commit**

```bash
git add Tools/CMakeLists.txt
git commit -m "build: replace tool boilerplate with twisty_add_tool"
```

---

### Task 5: Tests built everywhere + CTest registration

**Files:**
- Modify: `CMakeLists.txt`
- Rewrite: `Tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `Twisty` (Task 2 state), root `enable_testing()` (Task 1).
- Produces: CTest test `WeightTableTest`; `Tests/` builds on all platforms.

- [ ] **Step 1: Un-gate Tests in root `CMakeLists.txt`**

Change the enabled-test block from:

```cmake
enable_testing()
if(WIN32)
    add_subdirectory(Tests)
endif()
```

to:

```cmake
enable_testing()
add_subdirectory(Tests)
```

- [ ] **Step 2: Rewrite `Tests/CMakeLists.txt`** (full replacement)

```cmake
add_executable(WeightTableTest
    WeightTableTest.cpp
)

target_compile_features(WeightTableTest
    PUBLIC cxx_static_assert
    PUBLIC cxx_std_17
)

if(WIN32)
    target_compile_options(WeightTableTest
        PUBLIC "/openmp"
    )
endif()

target_link_libraries(WeightTableTest
    PUBLIC Twisty
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME WeightTableTest COMMAND WeightTableTest)
endif()
```

(Removed the now-redundant `PUBLIC glm` / `PRIVATE nlohmann_json::nlohmann_json` — both come transitively from `Twisty` PUBLIC. The test is pure in-memory computation, no data files.)

- [ ] **Step 3: Verify build + ctest**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: build succeeds including `WeightTableTest`; ctest runs it and reports 1/1 passed. If the table computation is slow (numStepsInt=20000), allow a generous timeout — do not lower the test's loop counts.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt Tests/CMakeLists.txt
git commit -m "build: build tests on all platforms and register with CTest"
```

---

### Task 6: Bootstrap script + README

**Files:**
- Create: `scripts/bootstrap-ubuntu.sh` (executable)
- Rewrite: `README.md`
- Delete: `install.txt`

**Interfaces:**
- Consumes: the dependency set from Task 2.
- Produces: a one-shot Ubuntu setup script; README documents configure/build/test; `install.txt` removed.

- [ ] **Step 1: Create `scripts/bootstrap-ubuntu.sh`**

```bash
#!/usr/bin/env bash
# One-shot dependency setup for building Twisty on Ubuntu.
set -euo pipefail

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenvdb-dev \
    libtbb-dev \
    libomp-dev \
    nlohmann-json3-dev \
    libboost-dev
```

Make it executable: `chmod +x scripts/bootstrap-ubuntu.sh`

Note: glm, FMath and stb are fetched by CMake itself (FMath's FetchContent pulls glm; `dependencies/CMakeLists.txt` fetches FMath + stb) — no apt package needed for them.

- [ ] **Step 2: Rewrite `README.md`** (full replacement — current file is only `# Twisty\nMultiple Scattering Research Codebase\n`)

```markdown
# Twisty

Multiple Scattering Research Codebase

## Building (Ubuntu)

```bash
./scripts/bootstrap-ubuntu.sh   # apt deps once per machine
cmake --preset linux            # configure into build/
cmake --build build             # build everything
ctest --test-dir build          # run the test suite
```

C++17 / CMake >= 3.20 / Ninja. OpenVDB, TBB, OpenMP, Boost and
nlohmann-json come from apt; FMath, stb and tinyexr are fetched by
CMake (FMath and stb via `dependencies/`, tinyexr via `Experiments/`).

Experiments and tools grow one line each through the
`twisty_add_experiment` / `twisty_add_tool` helpers in
`cmake/TwistyHelpers.cmake`.
```

- [ ] **Step 3: Delete `install.txt`**

```bash
git rm install.txt
```

- [ ] **Step 4: Verify**

```bash
test -x scripts/bootstrap-ubuntu.sh
cmake --preset linux && cmake --build build -j$(nproc) && ctest --test-dir build
```

Expected: all green. (Do not run the apt install on a machine that already has the deps; the script's idempotency isn't critical for this task's verification.)

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap-ubuntu.sh README.md
git commit -m "docs: add bootstrap script and rewrite build instructions"
```

---

### Task 7: Neural stub-out, stray experiments, submodule removal

**Files:**
- Delete: `Neural/CMakeLists.txt`, `Neural/CMakePresets.json`, `Neural/dependencies/CMakeLists.txt`
- Move: `Benchmark_5_1.cpp`, `DifferentNormalSThetaExperiment.cpp`, `FullExperiment_CombinedInitialCurves.cpp` → `Experiments/`
- Modify: `Experiments/CMakeLists.txt`
- Delete (gitlinks): `stb` (mode 160000), `tinyexr` (mode 160000, empty dir), `.gitmodules`

**Interfaces:**
- Consumes: `twisty_add_experiment` (Task 1), `stb_headers` via FetchContent (Task 2).
- Produces: no submodules; two new experiment targets (`Benchmark_5_1`, `DifferentNormalSThetaExperiment`); neural CMake scaffolding gone; `FullExperiment_CombinedInitialCurves.cpp` present but unbuilt.

- [ ] **Step 1: Remove Neural build scaffolding**

```bash
git rm Neural/CMakeLists.txt Neural/CMakePresets.json
git rm Neural/dependencies/CMakeLists.txt
rmdir Neural/dependencies   # now empty
```

Keep `Neural/*.cpp`, `Neural/python/`, `Neural/bunny*.vdb`, `Neural/test_sph.cpp`, `Neural/raymarch*.cpp`, any other source/data.

- [ ] **Step 2: Move the stray root experiments into `Experiments/`**

```bash
git mv Benchmark_5_1.cpp Experiments/Benchmark_5_1.cpp
git mv DifferentNormalSThetaExperiment.cpp Experiments/DifferentNormalSThetaExperiment.cpp
git mv FullExperiment_CombinedInitialCurves.cpp Experiments/FullExperiment_CombinedInitialCurves.cpp
```

- [ ] **Step 3: Add the two buildable strays to `Experiments/CMakeLists.txt`**

Append to the experiments section (both include only Twisty/FMath headers, plain flag; both live under `Experiments/` so the include paths resolve via `Twisty`'s PUBLIC source dir):

```cmake
twisty_add_experiment(Benchmark_5_1                         Benchmark_5_1.cpp)
twisty_add_experiment(DifferentNormalSThetaExperiment       DifferentNormalSThetaExperiment.cpp)
```

Do **not** add `FullExperiment_CombinedInitialCurves` — it includes `FullExperimentRunnerOptimalPerturbOptimized.h`, which does not exist (decision per spec; retrievable from git history).

- [ ] **Step 4: Remove submodules**

```bash
git rm stb
rm -rf stb
git rm tinyexr
git rm .gitmodules
```

Note: `tinyexr` is a stale empty gitlink with no `.gitmodules` entry (that's the `fatal: no submodule mapping` from `git submodule status`). `openvdb` was already removed from the index.

- [ ] **Step 5: Verify everything**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
git submodule status          # must print nothing (empty)
git ls-files -s | grep -E ' 160000 '   # must print nothing
git ls-files | grep -E '^(stb|tinyexr)/'   # must print nothing
test ! -e .gitmodules
```

Expected: configure+build green (stb now comes from FetchContent), ctest passes, zero gitlinks, no submodule directories tracked. Build also compiles the two new targets `Benchmark_5_1` and `DifferentNormalSThetaExperiment`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "build: stub out Neural, adopt stray experiments, drop submodules"
```

---

## Self-Review Notes

- Spec coverage: root config (Task 1), presets (Task 1), helpers (Task 1), dependency owner (Task 2), Boost declaration (Task 2), Experiments collapse (Task 3), Tools collapse (Task 4), CTest (Task 5), bootstrap/README/install.txt (Task 6), Neural (Task 7), stray files (Task 7), submodule removal incl. tinyexr gitlink (Task 7), `FullExperiment_CombinedInitialCurves` move-without-build (Task 7).
- Risk items tracked in Global Constraints: glm (no find_package), Boost fallback, stb tag pinned, tinyexr Populate warning accepted.
- Deviation flagged: EDGE-1 (uniform `/openmp` on WIN32) and the unconditional `RaycastVolume` target (already unconditional today; `# if(LINUX)` was commented out).