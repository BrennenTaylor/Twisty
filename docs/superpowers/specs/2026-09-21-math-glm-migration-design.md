# Design: Migrate Twisty Math from FMath (Farlor) to glm

Date: 2026-09-21
Status: Approved (in-chat design review, 2026-09-21)

## Background and Goal

Twisty's math was written against the author's own math library, FMath
(namespace `Farlor`, repo `https://github.com/BrennenTaylor/FMath`), fetched
via FetchContent in `dependencies/CMakeLists.txt`. The author has decided to
drop FMath and use glm (`https://github.com/g-truc/glm`) instead.

Impetus findings (audited during brainstorming):

- In the built tree (`Twisty/`, `Experiments/`, `Tools/` main, `Tests/`),
  `Farlor::` is used almost entirely as `Farlor::Vector3`: ~807 raw
  references across ~52 files. Real use of `Farlor::Vector4`,
  `Farlor::Vector2`, `Farlor::Matrix3x3`, `Quaternion`, `Ray`, `Plane`,
  `Line` in the built tree is zero (the only `Matrix3x3` hits are comments
  in the unbuilt `Twisty/FullExperimentRunnerOldMethodBridge.cpp`).
- FMath itself fetches GLM and Microsoft GSL for its internals. GSL has
  zero uses anywhere in the repo; it arrives only because FMath's
  CMakeLists links it. Three noise-circle experiments
  (`NoisyCirclePathGenerationMSegment{,Pinhole,RaycastVolume}.cpp`) already
  `#include <glm/...>` and consume the `glm` target FMath drags in.
- `Tests/` has zero Farlor usage.

Goal: single math library (glm), pinned and reproducible, with FMath and GSL
removed from the build entirely.

## Scope

In scope:

- `Twisty/` (22 files, 293 `Farlor::Vector3` refs)
- `Experiments/` active experiments (30 files, 434 refs)
- `Tools/` built executables (2 files, 6 refs)
- `dependencies/CMakeLists.txt`, `Twisty/CMakeLists.txt`,
  `Experiments/CMakeLists.txt` wiring
- `Tests/` (verification only)

Out of scope (leave untouched, do not migrate):

- `Viewer/` (Qt GUI, not built)
- `Neural/` (build scaffolding stubbed, sources kept, not built)
- `Tools/deprecated/` (not built)
- `RelatedWork/`, `unitweights/`
- The three moved-but-unbuilt strays in `Experiments/`
  (`Benchmark_5_1.cpp`, `DifferentNormalSThetaExperiment.cpp`,
  `FullExperiment_CombinedInitialCurves.cpp`)

Consequence to record and communicate: once the FMath fetch is removed, code
outside the built tree that still references `Farlor::...` will no longer
have the FMath headers available to it. This does not change their current
status — none of it builds today (several strays already reference the
deleted `FullExperimentRunner.h` and rearchitected APIs) — but it is a
documented effect of this change.

## Design

### Dependency wiring (Section 1 of approved design)

In `dependencies/CMakeLists.txt`:

- Remove the `FetchContent_Declare(FMath)` / `FetchContent_MakeAvailable(FMath)`
  block.
- Add, replacing it:

  ```cmake
  FetchContent_Declare(glm
      GIT_REPOSITORY "https://github.com/g-truc/glm.git"
      GIT_TAG "1.0.3"
  )
  FetchContent_MakeAvailable(glm)
  ```

- `glm` 1.0.3 (released 2025-12-31) builds no tests by default
  (`GLM_BUILD_TESTS` defaults OFF).
- `Twisty/CMakeLists.txt`: drop `PUBLIC FMath::FMath` from
  `target_link_libraries(Twisty ...)`; keep `PUBLIC glm`.
- `Experiments/CMakeLists.txt`: no change (`ExperimentBase` already links
  `glm` PUBLIC).
- GSL disappears automatically (it was linked privately by FMath only).
- No apt package changes (glm is header-only), no `find_package(glm)`, no
  `IMPORTED_GLOBAL` promotion entry. The `glm` target is defined in
  `dependencies/`, the same directory where it currently resolves for the
  sibling `Twisty/` and `Experiments/` directories; cross-directory
  visibility is reproduced exactly.
- This makes `glm` the singular math target; do not `find_package(glm)`
  anywhere.

### Type and API mapping (Section 2 of approved design)

Header swap:

- `<FMath/Vector3.h>` and `<FMath/FMath.h>` -> `<glm/glm.hpp>`
  (add `glm/gtc/matrix_transform.hpp` only where the mapping audit shows
  matrix helpers are genuinely used — currently only the noise-circle
  experiments already include it).

Type swap:

- `Farlor::Vector3` -> `glm::vec3` (the only type with real built-tree
  usage).

API mapping (audit counts from the built tree):

| Farlor call | glm replacement |
|---|---|
| `v.Dot(w)` (41 sites) | `glm::dot(v, w)` |
| `v.Cross(w)` (28) | `glm::cross(v, w)` |
| `v.Normalized()` (~119) | `glm::normalize(v)` |
| `v.Normalize()` (2, in-place void) | `v = glm::normalize(v)` |
| `v.Magnitude()` (46) | `glm::length(v)` |
| `v.SqrMagnitude()` (12) | `glm::length2(v)` |
| element-wise vec+vec / vec-scalar `+ - * /` | same semantics (glm is component-wise) |
| member `.x` `.y` `.z`, `v[i]` | identical |
| `==` / `!=` between vectors | use plain `==` if it resolves correctly, else `glm::all(glm::equal(a, b))` — decide by compile, keep consistent |
| `operator<<` streaming of vectors (few spots) | local helper overload (glm provides no stream operator) |

### Numerics and acceptance criteria (Section 3 of approved design)

User ruling: small numeric drift is acceptable.

How the WeightTableTest actually works (verified in
`Twisty/PathWeightUtils.cpp:123-158`, `203-220`): the cached table is keyed
by a UUID of the weighting parameters only. On a cold cache the constructor
computes the table with the current math, writes a fresh `.cwt`, then the
second table object in the test loads that same fresh file back and the
test compares computed-vs-serialized round-trip (equality of the same-code
output, not cross-library output). The test therefore does NOT, by itself,
prove pre-migration >= post-migration accuracy — that check must be an
explicit one-off diff harness.

- Farlor normalize is expected to divide by `sqrt(len2)`; glm typically
  multiplies by `1/sqrt(len2)`. Expect 1-ulp-class differences. Confirm by
  reading both implementations during implementation and record the
  observed drift.

Acceptance criteria:

1. Configure + full build green (`cmake --preset linux && cmake --build
   build`, ~97 targets).
2. WeightTableTest passes (1/1 via `ctest --test-dir build`).
   Precondition: preserve the pre-migration baseline first (below), then run
   the test on the migrated tree with a cleared
   `build/Tests/CachedWeightTables/` cache so the constructor actually
   computes with glm rather than loading the pre-existing `.cwt`.
3. One-off drift harness (scratch, not committed): preserve the Farlor
   cache created by a pre-migration WeightTableTest run (the `.cwt` plus
   its UUID export directory) to /tmp; after migration re-run the test to
   produce the glm cache; decode both `.cwt` files as their float arrays
   and report max absolute and max relative per-element deltas. Record in
   the implementation report. Escalate for review any relative delta
   above 1e-4 (a few-ulp argument is expected; larger drift implies a
   semantic difference in scaling, operators, or order of operations).
4. `Tests/` remains FMath-free; residual-scan greps (see Verification plan)
   come back clean.

### Branch and integration strategy

- New branch `math-glm-migration`, cut from `master` (which now contains
  the merged build-restructure, `b8f9e24`).
- Commit sequence follows the implementation plan (spec -> plan -> SDD
  execution). On completion, push the branch and open a PR against
  `master`, mirroring the restructure's PR flow.

## Verification plan

1. From scratch (or clean `/tmp` scratch build): configure + build
   `cmake --preset linux && cmake --build build`; 97 targets green.
2. `ctest --test-dir build`: 1/1 (WeightTableTest) passes with the
   tolerance/numerics criteria above.
3. Grep the built tree for residual references: no `#include <FMath`,
   `FMath::`, or `Farlor::` remaining in `Twisty/`, `Experiments/` (built),
   `Tools/` (built), `Tests/`.
4. `git grep -c FMath dependencies/CMakeLists.txt` -> 0.
5. Confirm `build/` links no `libFMath*` (via link.txt / build.ninja).
6. Record post-swap drift measurement for the weight tables.

## Out of scope (explicitly not done)

- Migrating unbuilt code (Viewer, Neural, Tools/deprecated, RelatedWork,
  unitweights, strays) to glm; they stay as-is.
- Modifying the FMath repository itself.
- Changing any numerical formulation of Twisty's physics beyond the
  mechanical Farlor->glm mapping above.