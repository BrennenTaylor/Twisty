#pragma once

#include "PathWeightUtils.h"

namespace twisty {
namespace PathWeighting {
    // Given a vector of curvatures, 1 per segement of a path, weight the path and return the long10 of the weight
    // TODO: We want to use span here, but currently not supported in compiler (is, but have to force latest verison).
    // Assumes that integral matches weighting params
    double WeightCurveViaCurvatureLog10(float *pCurvatureStart, uint32_t numCurvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &weightIntegral);

    __host__ __device__ double WeightCurveViaCurvatureLog10_CudaSafe(float *pCurvatureStart,
          uint32_t numCurvatures, const float *pWeightLookupTable,
          const int32_t weightLookupTableSize, const float ds, const float minCurvature,
          const float maxCurvature, const float curvatureStepSize);
}
}