#pragma once

#define _USE_MATH_DEFINES
#include "CurvePerturbUtils.h"

#include <FMath/FMath.h>

#include <random>

namespace twisty {
namespace PathGeneration {

    float CalculateMinimumArclength(twisty::PerturbUtils::BoundaryConditions boundaryConditions,
          uint32_t numSegmentsPerCurve);

    void ResolveTwoSegments(std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng);

    void ResolveThreeSegments(std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng);

    void ResolveEvenNumberOfSegments(const int numSegments, std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng);
}
}