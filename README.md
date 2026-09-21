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
