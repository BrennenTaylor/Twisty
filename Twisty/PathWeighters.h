#pragma once

#include "PathWeightUtils.h"

namespace twisty {
namespace PathWeighting {
    // Given a vector of curvatures, 1 per segement of a path, weight the path and return the long10 of the weight
    // TODO: We want to use span here, but currently not supported in compiler (is, but have to force latest verison).
    // Assumes that integral matches weighting params
    double WeightCurveViaCurvatureLog10(float *pCurvatureStart, uint32_t numCurvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &weightIntegral, const float absorption);

    __host__ __device__ double WeightCurveViaCurvatureLog10_CudaSafe(float *pCurvatureStart,
          uint32_t numCurvatures, const float *pWeightLookupTable,
          const int32_t weightLookupTableSize, const float ds, const float minCurvature,
          const float maxCurvature, const float curvatureStepSize, const float absorption);
}
}

#ifdef ACTUAL_IMPLEMENTATION

#include <stdint.h>

namespace twisty {
namespace PathWeighting {
    // Assume we have good pointers
    double WeightCurveViaCurvatureLog10(float *pCurvatureStart, uint32_t numCurvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &weightIntegral, const float absorption)
    {
        return WeightCurveViaCurvatureLog10_CudaSafe(pCurvatureStart, numCurvatures,
              weightIntegral.AccessLookupTable().data(),
              (uint32_t)weightIntegral.AccessLookupTable().size(), weightIntegral.GetDs(),
              weightIntegral.GetMinCurvature(), weightIntegral.GetMaxCurvature(),
              weightIntegral.GetCurvatureStepSize(), absorption);
    }

    __host__ __device__ double WeightCurveViaCurvatureLog10_CudaSafe(float *pCurvatureStart,
          uint32_t numCurvatures, const float *pWeightLookupTable,
          const int32_t weightLookupTableSize, const float ds, const float minCurvature,
          const float maxCurvature, const float curvatureStepSize, const float absorption)
    {
        if (!pCurvatureStart || (numCurvatures == 0)) {
            return 0.0;
        }

        // Currently assumes that we dont need a world space lookup
        const double absorptionFactor = std::exp(-absorption * ds);

        // Calculate value
        double runningPathWeightLog10 = 0.0;
        for (int segIdx = 0; segIdx < numCurvatures; ++segIdx) {
            // Extract curvature
            float curvature = pCurvatureStart[segIdx];

            if ((curvature < minCurvature) && abs(curvature - minCurvature) < 1e-3) {
                curvature = minCurvature;
            }

            if (curvature < minCurvature) {
                printf("Error: curvature less than min curvature: %f < %f\n", curvature,
                      minCurvature);
                printf("Forcing to min curvature\n");
                curvature = minCurvature;
            }

            if (curvature > maxCurvature) {
                printf("Error: curvature greater than max curvature: %f > %f\n", curvature,
                      maxCurvature);
                printf("Forcing to max curvature\n");
                curvature = maxCurvature;
            }

            float distanceFromMin = curvature - minCurvature;
            float realIdx = distanceFromMin / curvatureStepSize;
            int32_t leftIdx = (int32_t)floor(realIdx);
            int32_t rightIdx = leftIdx + 1;

            if (leftIdx == (weightLookupTableSize - 1)) {
                rightIdx--;  // Bump it left 1, it doesnt really matter anymore anyways.
            }

            float leftLookup = pWeightLookupTable[leftIdx];
            float rightLookup = pWeightLookupTable[rightIdx];

            float interpDist = distanceFromMin - (leftIdx * curvatureStepSize);

            double interpolatedResult
                  = leftLookup * (1.0f - interpDist) + (rightLookup * interpDist);

            // Adds in absorption per segment
            interpolatedResult *= absorptionFactor;

            // Take the natural log of the interpolated results
            double interpolatedResultLog10 = log10(interpolatedResult);
            if (isnan(interpolatedResultLog10)) {
                printf("Error: invalid segment weight, is nan\n");
                return 0.0;
            }

            // Update the running path weight. We also want to cache the segment weights
            runningPathWeightLog10 += interpolatedResultLog10;
        }
        // Factor in absorption

        if (isnan(runningPathWeightLog10)) {
            printf("Error: running path weight is nan\n");
            return 0.0;
        }
        return runningPathWeightLog10;
    }
}
}

#endif
