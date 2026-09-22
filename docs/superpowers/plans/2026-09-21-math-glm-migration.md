# FMath -> glm Math Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FMath (Farlor) math dependency with pinned glm across Twisty's built tree, while keeping results within tolerance and adding tests.

**Architecture:** Mechanically migrate `Farlor::Vector3` and its member API to `glm::vec3` and glm free functions in the built sources (Twisty/, Experiments/ built set, Tools/ built set), in three code-only tasks that keep the build green while FMath is still fetched; then swap the dependency wiring to a pinned glm 1.0.3 fetch and drop FMath + GSL; then lock the migration with a new test and validate numerics against the preserved pre-migration weight-table baseline. Unbuilt code (Viewer/, Neural/, Tools/deprecated/, strays, RelatedWork/, unitweights/) is left untouched.

**Tech Stack:** C++17, CMake 4.x + Ninja, FetchContent (pinned `glm` 1.0.3, no `find_package`), GLM header-only library, CTest.

**Spec:** `docs/superpowers/specs/2026-09-21-math-glm-migration-design.md` (approved 2026-09-21; the plan argues from the spec).

## Global Constraints

- glm is fetched via `FetchContent` pinned to `GIT_TAG "1.0.3"` (`https://github.com/g-truc/glm.git`). Do NOT `find_package(glm)` anywhere; glm is the singular math target.
- `Twisty/CMakeLists.txt`: remove `PUBLIC FMath::FMath`; KEEP `PUBLIC glm`.
- `Experiments/CMakeLists.txt`: no change.
- No apt package changes (glm is header-only); no new `IMPORTED_GLOBAL` promotion entry.
- Mapping is normative (see Migration Recipe); behavior must be preserved.
- Numerics: small drift acceptable; escalate for review any relative delta above **1e-4** in the Task 6 drift harness.
- Scope = built tree only: `Twisty/` (sources listed in `Twisty/CMakeLists.txt` `add_library(Twisty ...)`), `Experiments/` (sources named in `twisty_add_experiment(...)` calls + `ExperimentBase.cpp`), `Tools/` built executables (`Tools/*.cpp` added by `twisty_add_tool`), and `Tests/`. Everything else is out of scope and must not be edited.
- The three strays `Experiments/Benchmark_5_1.cpp`, `Experiments/DifferentNormalSThetaExperiment.cpp`, `Experiments/FullExperiment_CombinedInitialCurves.cpp` and `Twisty/FullExperimentRunnerOldMethodBridge.{h,cpp}` are NOT built; do not touch them.
- `Tests/` remains FMath-free. `Tests/CMakeLists.txt` may be edited to add the new test target.
- Preserved baseline (pre-migration, Farlor-computed weight tables): `/tmp/farlor-weighttables-baseline/` (10 `ds_*.csv` files + `9143073234636942748.cwt`). Do not delete it.
- Repo style: no code comments unless the surrounding file already documents the same construct; follow each file's local formatting.

---

## Migration Recipe (normative for Tasks 1-3)

Apply the following to EVERY migrated source file. Compile-driven fix loop; Ninja reports each remaining site as `file:line` errors.

### 1. Includes

| Before | After |
|---|---|
| `#include <FMath/Vector3.h>` | `#include <glm/glm.hpp>` |
| `#include "FMath/Vector3.h"` | `#include <glm/glm.hpp>` |
| `#include <FMath/FMath.h>` | `#include <glm/glm.hpp>` |
| `#include "FMath/FMath.h"` | `#include <glm/glm.hpp>` |
| `#include <FMath/Quaternion.h>` | `#include <glm/gtc/quaternion.hpp>` (check usage first; if unused after migration, omit) |
| any other `FMath/*.h` include | include the equivalent glm header by symbol; when uncertain, use the glm extension the symbol lives in, or `<glm/glm.hpp>` |

Leave commented-out includes (`// #include <FMath/...>`) untouched. If the file now has BOTH `<FMath...>` replaced and a duplicate produced, dedupe the include.

### 2. Type references

- `Farlor::Vector3` -> `glm::vec3` (exact string, all occurrences).
- If any `Farlor::` symbol other than `Vector3` survives in a migrated file (e.g. `Vector4`, `Matrix3x3`, `Quaternion`, `Ray`, `Plane`, free functions, `F_PI`), STOP and report it — none are expected in the built tree; a stray proves the file inventory is wrong.

### 3. Member-call mapping (receiver is a vec3)

| Before | After |
|---|---|
| `v.Dot(w)` | `glm::dot(v, w)` |
| `v.Cross(w)` | `glm::cross(v, w)` |
| `v.Normalized()` | `glm::normalize(v)` |
| `v.Normalize();` (void, in place) | `v = glm::normalize(v);` |
| `v.Magnitude()` | `glm::length(v)` |
| `v.SqrMagnitude()` | `glm::length2(v)` |

When a call's result feeds another expression, preserve the expression semantics exactly (e.g. `float n = a.Dot(a);` -> `float n = glm::dot(a, a);`).

### 4. Operators

- Component-wise `+ - * /` between vec3 and vec3, and between vec3 and scalar: unchanged — glm has identical semantics.
- `operator[]`: unchanged.
- Member `.x .y .z`: unchanged.
- **Vector equality / inequality** (either operand is a vec3): rewrite to the bool form:
  - `a == b` -> `glm::all(glm::equal(a, b))`
  - `a != b` -> `!glm::all(glm::equal(a, b))`
  - Scalar comparisons (`float == float`) stay as `==`.
- **Streaming a vec3 with `<<`**: glm has no `operator<<`. At each site, add `#define GLM_ENABLE_EXPERIMENTAL` (before any glm include in that TU) and `#include <glm/gtx/string_cast.hpp>`, then emit `glm::to_string(v)` (returns `std::string`). If the surrounding stream op is `<<` on a custom struct that now streams a vec3, keep the struct's `<<` and forward to `glm::to_string`.

### 5. Constructors

- `Farlor::Vector3()` -> `glm::vec3()` (zero).
- `Farlor::Vector3(x, y, z)` -> `glm::vec3(x, y, z)`.
- `Farlor::Vector3(s)` (single scalar) -> `glm::vec3(s)`.
- If a variable was default-constructed then assigned component-wise, leave as-is (glm default-constructs to zero).

### 6. Verify for a migrated task

Commands (run from repo root; do NOT clear the preserved baseline in `/tmp`):

```bash
cmake --preset linux
cmake --build build -j$(nproc)
ctest --test-dir build
```

Residual grep — no output expected when scoped to the task's built files:

```bash
git grep -nE "Farlor::|#include [<\"](FMath|Farlor)" -- <task's built files>
```

---

## File Structure

- `dependencies/CMakeLists.txt` — replaces the FMath FetchContent block with the pinned glm block (Task 4).
- `Twisty/CMakeLists.txt` — drops `PUBLIC FMath::FMath` (Task 4).
- `Twisty/*.{h,cpp}` (sources in the `add_library(Twisty ...)` list) — glm migration (Task 1).
- `Experiments/*.cpp` (files named in `twisty_add_experiment(...)` calls + `ExperimentBase.cpp`) — glm migration (Task 2).
- `Tools/*.cpp` (built exes) — glm migration (Task 3).
- `Tests/MathMigrationTest.cpp` + `Tests/CMakeLists.txt` — new test (Task 5).
- `README.md` — dependency list wording (Task 7).

---

### Task 1: Migrate Twisty/ core library to glm

**Files:**
- Modify: every `Twisty/*.{h,cpp}` that appears in the `add_library(Twisty ...)` list AND contains `Farlor::` or `#include <FMath...>` / `#include "FMath..."`
  - Derive the inventory by reading `Twisty/CMakeLists.txt` `add_library(Twisty ...)` source list, then `grep -rlE "Farlor::|#include [<\"](FMath|Farlor)" Twisty --include=*.h --include=*.cpp` filtered to that list.
  - Do NOT touch `Twisty/FullExperimentRunnerOldMethodBridge.{h,cpp}` (not in the list; unbuilt).

**Interfaces:**
- Consumes: nothing new — FMath is still fetched in this task, glm is still available transitively.
- Produces: `Twisty/` sources compile with glm and no `Farlor::`/FMath includes; downstream tasks build on the migrated headers.

- [ ] **Step 1: Inventory the built Twisty sources**
  - From `Twisty/CMakeLists.txt` extract the `add_library(Twisty ...)` payload (extensions `.cpp` and `.h`). Cross it with the grep for `Farlor::|#include [<\"](FMath|Farlor)`. Write the file list into the task report.
  - Expected scale: ~20 sources under `Twisty/`.

- [ ] **Step 2: Apply the Migration Recipe to every inventory file**
  - Includes, type refs, member calls, operators, constructors per the Recipe.

- [ ] **Step 3: Build and fix until green**

```bash
cmake --build build -j$(nproc)
```

  Expected: compiler errors name remaining `Farlor::`/member-call sites; fix each per the Recipe, re-run until green. Watch for: `.Transform(`/`Angle(` (NOT expected; report if seen, do not guess), vector `==`, and stream `<<` sites.

- [ ] **Step 4: Confirm tests still run**

```bash
ctest --test-dir build
```

  Expected: 1/1 pass (WeightTableTest). After Task 1 the cached table still matches (no numeric change yet, or the test round-trips; Task 6 owns numeric proof).

- [ ] **Step 5: Residual grep on the migrated inventory**

```bash
git grep -nE "Farlor::|#include [<\"](FMath|Farlor)" -- $(cat inventory)
```

  Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add Twisty
git commit -m "refactor: migrate Twisty core math from Farlor to glm"
```

---

### Task 2: Migrate the built Experiments to glm

**Files:**
- Modify: `Experiments/ExperimentBase.cpp` and every `.cpp` named in a `twisty_add_experiment(...)` call in `Experiments/CMakeLists.txt` that contains `Farlor::` or an FMath include.
  - Derive the inventory from `Experiments/CMakeLists.txt`; exclude the three strays (`Benchmark_5_1.cpp`, `DifferentNormalSThetaExperiment.cpp`, `FullExperiment_CombinedInitialCurves.cpp`).
  - Expected scale: ~28 sources (incl. `ExperimentBase.cpp`). The noise-circle files already include `<glm/glm.hpp>`; migrate only their `Farlor` usage, leave their glm camera code alone.

**Interfaces:**
- Consumes: the migrated `Twisty/` headers (Task 1).
- Produces: built experiment sources compile with glm only.

- [ ] **Step 1: Inventory**
  - Extract the `.cpp` names from `twisty_add_experiment(...)` calls plus `ExperimentBase.cpp`; cross with the Farlor grep; exclude the strays. Write the list into the report.

- [ ] **Step 2: Apply the Migration Recipe to every inventory file**

- [ ] **Step 3: Build and fix until green**

```bash
cmake --build build -j$(nproc)
```

- [ ] **Step 4: Confirm tests still run**

```bash
ctest --test-dir build
```

  Expected: 1/1 pass.

- [ ] **Step 5: Residual grep on the migrated inventory**

```bash
git grep -nE "Farlor::|#include [<\"](FMath|Farlor)" -- $(cat inventory)
```

  Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add Experiments
git commit -m "refactor: migrate built experiments math from Farlor to glm"
```

---

### Task 3: Migrate the built Tools and run the whole-tree residual sweep

**Files:**
- Modify: `Tools/*.cpp` built by `twisty_add_tool(...)` in `Tools/CMakeLists.txt` that contain Farlor usage (expected: `Tools/ExportedFiveSegmentsFixer.cpp`, `Tools/ExportedPathFixer.cpp`).
- Test: whole built-tree residual graft.

**Interfaces:**
- Consumes: `Twisty/` + `Experiments/` migrated code (Tasks 1-2).
- Produces: a built tree free of Farlor/FMath references.

- [ ] **Step 1: Inventory and apply the Migration Recipe** to the Tools files.

- [ ] **Step 2: Build and fix until green**

```bash
cmake --build build -j$(nproc)
```

- [ ] **Step 3: Whole built-tree residual sweep** — all three commands must print nothing:

```bash
git grep -nE "Farlor::|#include [<\"](FMath|Farlor)" -- Twisty Experiments Tools Tests | grep -v -E "FullExperimentRunnerOldMethodBridge|Benchmark_5_1|DifferentNormalSThetaExperiment|FullExperiment_CombinedInitialCurves"
git grep -n "Farlor" -- Twisty/CMakeLists.txt Experiments/CMakeLists.txt Tools/CMakeLists.txt dependencies/CMakeLists.txt
```

  Note: commented `// #include <FMath/...>` lines and the four excluded unbuilt files are the ONLY allowed residual matches. If anything else appears, fix and re-grep.

- [ ] **Step 4: Confirm tests pass**

```bash
ctest --test-dir build
```

  Expected: 1/1 pass.

- [ ] **Step 5: Commit**

```bash
git add Tools
git commit -m "refactor: migrate built tool executables math from Farlor to glm"
```

---

### Task 4: Swap dependency wiring — pinned glm, drop FMath and GSL

**Files:**
- Modify: `dependencies/CMakeLists.txt` (remove the FMath FetchContent block; add the pinned glm block).
- Modify: `Twisty/CMakeLists.txt` (remove `PUBLIC FMath::FMath`).

**Interfaces:**
- Consumes: nothing FMath — all migrated code (Tasks 1-3) now uses glm only.
- Produces: the single pinned `glm` target; no `FMath`/`GSL` anywhere in the build.

- [ ] **Step 1: Edit `dependencies/CMakeLists.txt`**
  - Delete the block:
    ```cmake
    FetchContent_Declare(FMath
        GIT_REPOSITORY "https://github.com/BrennenTaylor/FMath"
    )
    FetchContent_MakeAvailable(FMath)
    ```
  - Add, in the version-controlled section:
    ```cmake
    FetchContent_Declare(glm
        GIT_REPOSITORY "https://github.com/g-truc/glm.git"
        GIT_TAG "1.0.3"
    )
    FetchContent_MakeAvailable(glm)
    ```

- [ ] **Step 2: Edit `Twisty/CMakeLists.txt`**
  - Remove the line `    PUBLIC FMath::FMath` from `target_link_libraries(Twisty ...)`. Keep `PUBLIC glm` and the other entries.

- [ ] **Step 3: Reconfigure and build**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
```

  Expected: green. Ninja drops FMath-compiled objects; `glm` now comes from our pinned fetch (the existing `build/_deps/glm-*` content is reused).

- [ ] **Step 4: Remove stale FMath fetch artifacts and re-verify**

```bash
rm -rf build/_deps/fmath-build build/_deps/fmath-src build/_deps/fmath-subbuild
cmake --preset linux && cmake --build build -j$(nproc)
```

  Expected: green, and no FMath re-fetch (nothing references it).

- [ ] **Step 5: From-scratch scratch build (fresh FetchContent) in /tmp**

```bash
cmake -S . -B /tmp/twisty-glm-scratch -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/twisty-glm-scratch -j$(nproc)
```

  Expected: all ~97 targets build from a clean configure (FMath and GSL absent from the dependency graph).

- [ ] **Step 6: Tests + residual checks**

```bash
ctest --test-dir build
```

  Expected: 1/1 pass. Then:
  - `grep -c Farlor dependencies/CMakeLists.txt` -> 0
  - `grep -rio "fmath" build/build.ninja` -> no output
  - `ls build/_deps | grep -i fmath` -> no output

- [ ] **Step 7: Commit**

```bash
git add dependencies/CMakeLists.txt Twisty/CMakeLists.txt
git commit -m "build: pin glm 1.0.3 via FetchContent, drop FMath dependency"
```

---

### Task 5: Add MathMigrationTest to lock the migrated math contract

**Files:**
- Create: `Tests/MathMigrationTest.cpp`
- Modify: `Tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the pinned `glm` target (from Task 4) — the test links glm only; it does NOT need Twisty.
- Produces: a fast deterministic CTest target proving the migrated glm primitives produce the exact values the migrated code relies on.

- [ ] **Step 1: Write the test**

Write `Tests/MathMigrationTest.cpp`:

```cpp
#include <cmath>
#include <iostream>

#include <glm/glm.hpp>

static bool nearlyEqual(float a, float b)
{
    return std::abs(a - b) < 1e-6f;
}

int main()
{
    const glm::vec3 a(1.0f, 2.0f, 3.0f);
    const glm::vec3 b(4.0f, -5.0f, 6.0f);

    // Dot — declared 1*4 + 2*(-5) + 3*6
    if (!nearlyEqual(glm::dot(a, b), 12.0f)) {
        std::cout << "FAILURE: glm::dot" << std::endl;
        return 1;
    }

    // Cross — (2*6 - 3*(-5), 3*4 - 1*6, 1*(-5) - 2*4)
    const glm::vec3 c = glm::cross(a, b);
    if (!nearlyEqual(c.x, 27.0f) || !nearlyEqual(c.y, 6.0f) || !nearlyEqual(c.z, -13.0f)) {
        std::cout << "FAILURE: glm::cross" << std::endl;
        return 1;
    }

    // Length / length2
    if (!nearlyEqual(glm::length(a), std::sqrt(14.0f)) || !nearlyEqual(glm::length2(a), 14.0f)) {
        std::cout << "FAILURE: glm::length/length2" << std::endl;
        return 1;
    }

    // Normalize — unit vector on the exact same line
    const glm::vec3 n = glm::normalize(a);
    if (!nearlyEqual(glm::length(n), 1.0f)) {
        std::cout << "FAILURE: glm::normalize" << std::endl;
        return 1;
    }

    // Component-wise + and scalar *
    const glm::vec3 sum = a + b;
    const glm::vec3 scaled = a * 2.0f;
    if (!nearlyEqual(sum.x, 5.0f) || !nearlyEqual(sum.y, -3.0f) || !nearlyEqual(sum.z, 9.0f)
        || !nearlyEqual(scaled.x, 2.0f) || !nearlyEqual(scaled.z, 6.0f)) {
        std::cout << "FAILURE: component-wise + / scalar *" << std::endl;
        return 1;
    }

    // Vector equality via glm::all(glm::equal(...))
    if (!glm::all(glm::equal(a, a)) || glm::all(glm::equal(a, b))) {
        std::cout << "FAILURE: vector equality" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: MathMigrationTest passes!" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Register with CTest** — edit `Tests/CMakeLists.txt`, inside the existing `if(BUILD_TESTING)` block, after the WeightTableTest registration:

```cmake
add_executable(MathMigrationTest MathMigrationTest.cpp)
target_link_libraries(MathMigrationTest PRIVATE glm)
add_test(NAME MathMigrationTest COMMAND MathMigrationTest)
```

- [ ] **Step 3: Build and run the new test**

```bash
cmake --preset linux && cmake --build build -j$(nproc)
ctest --test-dir build
```

  Expected: **2/2** passing (WeightTableTest + MathMigrationTest), MathMigrationTest near-instant.

- [ ] **Step 4: Confirm the test target is FMath-free**

```bash
git grep -c "FMath\|Farlor" -- Tests/MathMigrationTest.cpp
```

  Expected: 0.

- [ ] **Step 5: Commit**

```bash
git add Tests/MathMigrationTest.cpp Tests/CMakeLists.txt
git commit -m "test: add MathMigrationTest locking the glm math contract"
```

---

### Task 6: Numeric drift validation (scratch harness — NOT committed)

**Files:**
- Reference (read-only): `/tmp/farlor-weighttables-baseline/` (preserved pre-migration Farlor tables).
- Create (in `/tmp`, never in the repo): `diff_weighttables.py`.

**Interfaces:**
- Consumes: the migrated build (Tasks 1-5) and the preserved baseline.
- Produces: a recorded max absolute / relative per-element drift figure and a pass/fail verdict against the 1e-4 relative bar.

- [ ] **Step 1: Regenerate the weight tables with glm math**

```bash
rm -rf build/Tests/CachedWeightTables
ctest --test-dir build
```

  Expected: cold run (~17-53s) recomputes with glm and rewrites the cache under `build/Tests/CachedWeightTables/`.

- [ ] **Step 2: Read both normalization implementations (spec acceptance item)**

  - Read `build/_deps/fmath-src/FMath/Vector3.cpp` `Normalized()`/`Normalize()` and the glm implementation in `build/_deps/glm-src/glm/vector_relational.inl`/wherever `normalize` lives for vec3 (commonly `glm/simd/` or `glm/detail/func_geometric.inl`; it uses `meta::sqrt` and a 1/sqrt multiply). Quote both in the report to document the expected 1-ulp-class difference.

- [ ] **Step 3: Write the diff harness** (scratch only)

Write `/tmp/diff_weighttables.py`:

```python
import csv, sys
import numpy as np

def load(dirpath):
    # Each ds_*.csv is headerless "index, value" floats. Match by file name.
    out = {}
    for p in sorted(__import__('glob').glob(dirpath + '/*.csv')):
        rows = list(csv.reader(open(p, newline='')))
        vals = np.array([float(row[1]) for row in rows], dtype=np.float64)
        out[__import__('os').path.basename(p)] = vals
    return out

base = load('/tmp/farlor-weighttables-baseline/9143073234636942748')
new = load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/farlor-weighttables-baseline/9143073234636942748')

assert set(base) == set(new), f'key mismatch: {set(base) ^ set(new)}'
worst_abs, worst_rel = 0.0, 0.0
for k in base:
    d = np.abs(base[k] - new[k])
    rel = np.divide(d, np.abs(base[k]), out=np.zeros_like(d), where=np.abs(base[k]) > 0)
    worst_abs = max(worst_abs, float(d.max()))
    worst_rel = max(worst_rel, float(rel.max()))
print(f'max_abs_delta = {worst_abs:.6e}')
print(f'max_rel_delta = {worst_rel:.6e}')
print('PASS' if worst_rel <= 1e-4 else 'REVIEW')
```

  Note: if the new run produced a different export-directory name than `9143073234636942748`, pass the new directory path as the argument (the script also matches by sorted file name).

- [ ] **Step 4: Run the harness**

```bash
python3 /tmp/diff_weighttables.py
```

  Expected: prints `max_abs_delta`, `max_rel_delta`, and `PASS` (worst_rel <= 1e-4). Record the numbers verbatim in the report.

- [ ] **Step 5: Record verdict in the SDD ledger**

  - If PASS: acceptance criterion 3 met; note that the gitignored caches are regenerated with glm math and the ramp of drift is documented.
  - If REVIEW: STOP and report — do not proceed; a relative delta above 1e-4 implies a semantic difference (scaling, operator, or order of operations) that must be investigated before merging.

(This task touches no committed files; nothing to commit.)

---

### Task 7: Update build docs for the glm dependency

**Files:**
- Modify: `README.md` (lines ~15-16, the dependency-listing paragraph).

**Interfaces:**
- Consumes: nothing.
- Produces: documentation that names glm, not FMath.

- [ ] **Step 1: Read the current wording**

```bash
sed -n '10,20p' README.md
```

- [ ] **Step 2: Edit the dependency sentence**

  Replace the sentence listing FMath among the fetched deps (currently: "FMath, stb and tinyexr are fetched by CMake (FMath and stb via `dependencies/`, tinyexr via `Experiments/`)") with a glm-equivalent, e.g.:

  > ...nlohmann-json come from apt; glm, stb and tinyexr are fetched by CMake (glm and stb via `dependencies/`, tinyexr via `Experiments/`). glm is pinned to 1.0.3.

  Preserve the surrounding prose and the README's tone.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document glm 1.0.3 as the math dependency"
```

---

## Self-Review Notes

- Spec coverage: dependency wiring = Task 4; type/API mapping = Tasks 1-3; numerics/acceptance (drift harness, 1e-4 bar, baseline preservation) = Task 6; tests = Task 5; docs = Task 7; "Tests/ remains FMath-free" = Tasks 3 + 5 residual greps; leave-unbuilt = explicit exclusions in every inventory step. No spec section lacks a task.
- Placeholders: none — every step carries exact commands, code, or explicit grep expectations; the only "expected ~N" are scale estimates with a derivation method.
- Type/name consistency: `glm::vec3`, `glm::dot/cross/normalize/length/length2`, `glm::all(glm::equal(...))`, and the `1.0.3` pin are the same strings in the Recipe, tests, and dependency block. Stray symbols must be reported, not guessed (Recipe section 2).
- Execution ordering is load-bearing: Tasks 1-3 keep FMath fetched so the tree stays green during code migration; Task 4 flips wiring; Task 5 registers the new test against the post-Task-4 `glm` target; Task 6 measures drift last.