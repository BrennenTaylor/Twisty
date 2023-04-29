#pragma once

#include "PathWeightUtils.h"
#include "boost/math/special_functions/math_fwd.hpp"

#ifdef __linux__
#include <openvdb/openvdb.h>
#endif

namespace twisty {
namespace PathWeighting {

    struct PathWeightValue {
        bool isValid = false;
        double weight = 0.0f;
    };

    // Given a vector of curvatures, 1 per segement of a path, weight the path and return the long10 of the weight
    // TODO: We want to use span here, but currently not supported in compiler (is, but have to force latest verison).
    // Assumes that integral matches weighting params
    PathWeightValue WeightCurveViaCurvatureLog10(float *pCurvatureStart, uint32_t numCurvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &weightIntegral,
          const float absorption);

    __host__ __device__ PathWeightValue WeightCurveViaCurvatureLog10_CudaSafe(
          float *pCurvatureStart, uint32_t numCurvatures, const float *pWeightLookupTable,
          const int32_t weightLookupTableSize, const float ds, const float minCurvature,
          const float maxCurvature, const float curvatureStepSize, const float absorption);

    PathWeightValue WeightCurveViaPositionLog10_PositionDependent(
          const std::vector<Farlor::Vector3> &positions, const std::vector<float> &curvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &environmentLookupTable,
          const twisty::PathWeighting::BaseWeightLookupTable &objectLookupTable,
          const float environmentAbsorbtion);

#ifdef __linux__
    PathWeightValue WeightCurveViaPositionLog10_PositionDependent(
          const std::vector<Farlor::Vector3> &positions, const std::vector<float> &curvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &environmentLookupTable,
          const twisty::PathWeighting::BaseWeightLookupTable &objectLookupTable,
          const float environmentAbsorbtion, openvdb::FloatGrid::Ptr grid);
#endif
}
}